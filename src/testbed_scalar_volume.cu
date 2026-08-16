/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   testbed_scalar_volume.cu
 *  @brief  Fits a dense uint8 intensity volume, f(x in [0,1]^3) -> intensity.
 */

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/common_host.h>
#include <neural-graphics-primitives/random_val.cuh>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/thread_pool.h>

#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/trainer.h>

#include <json/json.hpp>

#include <blosc.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

using json = nlohmann::json;

namespace ngp {

Testbed::NetworkDims Testbed::network_dims_scalar_volume() const {
	NetworkDims dims;
	dims.n_input = 3;
	dims.n_output = 1;
	dims.n_pos = 3;
	return dims;
}

__global__ void eval_scalar_volume_kernel(
	uint32_t n_elements, const uint8_t* __restrict__ data, const vec3* __restrict__ positions, ivec3 resolution, float* __restrict__ result
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) {
		return;
	}

	auto read_val = [&](int x, int y, int z) {
		return (float)data[((size_t)z * resolution.y + y) * resolution.x + x] * (1.0f / 255.0f);
	};

	vec3 pos = clamp(positions[i] * vec3(resolution) - 0.5f, 0.0f, vec3(resolution) - (1.0f + 1e-4f));

	const ivec3 pos_int = pos;
	const vec3 weight = pos - vec3(pos_int);

	const ivec3 idx = clamp(pos_int, 0, resolution - 2);

	result[i] = (1 - weight.x) * (1 - weight.y) * (1 - weight.z) * read_val(idx.x, idx.y, idx.z) +
		(weight.x) * (1 - weight.y) * (1 - weight.z) * read_val(idx.x + 1, idx.y, idx.z) +
		(1 - weight.x) * (weight.y) * (1 - weight.z) * read_val(idx.x, idx.y + 1, idx.z) +
		(weight.x) * (weight.y) * (1 - weight.z) * read_val(idx.x + 1, idx.y + 1, idx.z) +
		(1 - weight.x) * (1 - weight.y) * (weight.z) * read_val(idx.x, idx.y, idx.z + 1) +
		(weight.x) * (1 - weight.y) * (weight.z) * read_val(idx.x + 1, idx.y, idx.z + 1) +
		(1 - weight.x) * (weight.y) * (weight.z) * read_val(idx.x, idx.y + 1, idx.z + 1) +
		(weight.x) * (weight.y) * (weight.z) * read_val(idx.x + 1, idx.y + 1, idx.z + 1);
}

void Testbed::train_scalar_volume(size_t target_batch_size, bool get_loss_scalar, cudaStream_t stream) {
	const uint32_t n_output_dims = 1;
	const uint32_t n_input_dims = 3;

	const uint32_t batch_size = (uint32_t)target_batch_size;

	m_scalar_volume.training.positions.enlarge(batch_size);
	m_scalar_volume.training.targets.enlarge(batch_size);

	generate_random_uniform<float>(stream, m_rng, batch_size * n_input_dims, (float*)m_scalar_volume.training.positions.data());

	linear_kernel(
		eval_scalar_volume_kernel,
		0,
		stream,
		batch_size,
		m_scalar_volume.data.data(),
		m_scalar_volume.training.positions.data(),
		m_scalar_volume.resolution,
		m_scalar_volume.training.targets.data()
	);

	GPUMatrix<float> training_batch_matrix((float*)m_scalar_volume.training.positions.data(), n_input_dims, batch_size);
	GPUMatrix<float> training_target_matrix(m_scalar_volume.training.targets.data(), n_output_dims, batch_size);

	auto ctx = m_trainer->training_step(stream, training_batch_matrix, training_target_matrix);
	if (get_loss_scalar) {
		m_loss_scalar.update(m_trainer->loss(stream, *ctx));
	}

	m_training_step++;
}

__global__ void init_scalar_volume_coords(
	uint32_t sample_index,
	vec3* __restrict__ positions,
	float* __restrict__ depth_buffer,
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera_matrix,
	vec2 screen_center,
	vec3 parallax_shift,
	bool snap_to_pixel_centers,
	float plane_z,
	Foveation foveation,
	Buffer2DView<const uint8_t> hidden_area_mask,
	Lens lens
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	Ray ray = pixel_to_ray(
		sample_index,
		{(int)x, (int)y},
		resolution,
		focal_length,
		camera_matrix,
		screen_center,
		parallax_shift,
		snap_to_pixel_centers,
		0.0f, // near distance
		plane_z,
		0.0f, // aperture size
		foveation,
		hidden_area_mask,
		lens
	);

	uint32_t idx = x + resolution.x * y;
	if (!ray.is_valid()) {
		depth_buffer[idx] = MAX_DEPTH();
		positions[idx] = vec3(-1.0f);
		return;
	}

	// plane_z is negative, so this lands on the slicing plane in front of the camera.
	positions[idx] = ray.o - plane_z * ray.d;
	depth_buffer[idx] = -plane_z;
}

__global__ void shade_kernel_scalar_volume(
	ivec2 resolution, const vec3* __restrict__ positions, const float* __restrict__ values, vec4* __restrict__ frame_buffer
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	uint32_t idx = x + resolution.x * y;

	const vec3 pos = positions[idx];
	if (pos.x < 0.0f || pos.x > 1.0f || pos.y < 0.0f || pos.y > 1.0f || pos.z < 0.0f || pos.z > 1.0f) {
		frame_buffer[idx] = vec4(0.0f);
		return;
	}

	const vec3 color = srgb_to_linear(vec3(clamp(values[idx], 0.0f, 1.0f)));
	frame_buffer[idx] = {color.x, color.y, color.z, 1.0f};
}

void Testbed::render_scalar_volume(
	cudaStream_t stream,
	const CudaRenderBufferView& render_buffer,
	const vec2& focal_length,
	const mat4x3& camera_matrix,
	const vec2& screen_center,
	const Foveation& foveation,
	const Lens& lens
) {
	auto res = render_buffer.resolution;

	size_t n_pixels = (size_t)res.x * res.y;
	uint32_t n_elements = next_multiple((uint32_t)n_pixels, BATCH_SIZE_GRANULARITY);
	m_scalar_volume.render_coords.enlarge(n_elements);
	m_scalar_volume.render_out.enlarge(n_elements);

	// The kernel below only writes the first res.x*res.y coords, but the network is evaluated on the padded batch.
	CUDA_CHECK_THROW(cudaMemsetAsync(m_scalar_volume.render_coords.data(), 0, m_scalar_volume.render_coords.bytes(), stream));

	const float plane_z = -(m_slice_plane_z + m_scale);

	const dim3 threads = {16, 8, 1};
	const dim3 blocks = {div_round_up((uint32_t)res.x, threads.x), div_round_up((uint32_t)res.y, threads.y), 1};
	init_scalar_volume_coords<<<blocks, threads, 0, stream>>>(
		render_buffer.spp,
		m_scalar_volume.render_coords.data(),
		render_buffer.depth_buffer,
		res,
		focal_length,
		camera_matrix,
		screen_center,
		m_parallax_shift,
		m_snap_to_pixel_centers,
		plane_z,
		foveation,
		render_buffer.hidden_area_mask ? render_buffer.hidden_area_mask->const_view() : Buffer2DView<const uint8_t>{},
		lens
	);

	if (m_render_ground_truth) {
		linear_kernel(
			eval_scalar_volume_kernel,
			0,
			stream,
			n_elements,
			m_scalar_volume.data.data(),
			m_scalar_volume.render_coords.data(),
			m_scalar_volume.resolution,
			m_scalar_volume.render_out.data()
		);
	} else {
		GPUMatrix<float> positions_matrix((float*)m_scalar_volume.render_coords.data(), 3, n_elements);
		GPUMatrix<float, RM> values_matrix(m_scalar_volume.render_out.data(), 1, n_elements);
		m_network->inference(stream, positions_matrix, values_matrix);
	}

	shade_kernel_scalar_volume<<<blocks, threads, 0, stream>>>(
		res, m_scalar_volume.render_coords.data(), m_scalar_volume.render_out.data(), render_buffer.frame_buffer
	);
}

void Testbed::load_scalar_volume(const fs::path& data_path) {
	auto start = std::chrono::steady_clock::now();

	const fs::path level_path = data_path / "0";
	const fs::path zarray_path = level_path / ".zarray";
	if (!zarray_path.exists()) {
		throw std::runtime_error{fmt::format("Zarr metadata '{}' does not exist.", zarray_path.str())};
	}

	std::ifstream metadata_file{native_string(zarray_path)};
	const json metadata = json::parse(metadata_file, nullptr, true, true);

	const std::string dtype = metadata.value("dtype", std::string{});
	if (dtype != "|u1") {
		throw std::runtime_error{fmt::format("ScalarVolume requires the zarr dtype '|u1', but got '{}'.", dtype)};
	}

	const std::vector<int> shape = metadata.at("shape");
	const std::vector<int> chunks = metadata.at("chunks");
	if (shape.size() != 3 || chunks.size() != 3) {
		throw std::runtime_error{"ScalarVolume requires a 3-dimensional zarr array."};
	}

	bool is_blosc_compressed = false;
	if (metadata.contains("compressor") && !metadata["compressor"].is_null()) {
		const std::string compressor_id = metadata["compressor"].value("id", std::string{});
		if (compressor_id != "blosc") {
			throw std::runtime_error{fmt::format("Unsupported zarr compressor '{}'.", compressor_id)};
		}

		is_blosc_compressed = true;
	}

	const std::string separator = metadata.value("dimension_separator", std::string{"."});
	const bool is_nested = separator == "/";

	// zarr axis order is z, y, x; everything below is in x, y, z order.
	const ivec3 volume_shape = {shape[2], shape[1], shape[0]};
	const ivec3 chunk_size = {chunks[2], chunks[1], chunks[0]};
	const ivec3 n_chunks = {
		div_round_up(volume_shape.x, chunk_size.x), div_round_up(volume_shape.y, chunk_size.y), div_round_up(volume_shape.z, chunk_size.z)
	};

	auto chunk_path = [&](const ivec3& c) {
		if (is_nested) {
			return level_path / std::to_string(c.z) / std::to_string(c.y) / std::to_string(c.x);
		}

		return level_path / fmt::format("{}{}{}{}{}", c.z, separator, c.y, separator, c.x);
	};

	std::vector<ivec3> present_chunks;
	for (int cz = 0; cz < n_chunks.z; ++cz) {
		if (is_nested && !(level_path / std::to_string(cz)).exists()) {
			continue;
		}

		for (int cy = 0; cy < n_chunks.y; ++cy) {
			if (is_nested && !(level_path / std::to_string(cz) / std::to_string(cy)).exists()) {
				continue;
			}

			for (int cx = 0; cx < n_chunks.x; ++cx) {
				if (chunk_path({cx, cy, cz}).exists()) {
					present_chunks.emplace_back(cx, cy, cz);
				}
			}
		}
	}

	if (present_chunks.empty()) {
		throw std::runtime_error{fmt::format("No zarr chunks found in '{}'.", level_path.str())};
	}

	ivec3 chunk_min = present_chunks.front();
	ivec3 chunk_max = present_chunks.front();
	for (const auto& c : present_chunks) {
		chunk_min = min(chunk_min, c);
		chunk_max = max(chunk_max, c);
	}

	const ivec3 begin = chunk_min * chunk_size;
	const ivec3 end = min((chunk_max + 1) * chunk_size, volume_shape);
	const ivec3 resolution = end - begin;

	const size_t n_voxels = (size_t)resolution.x * resolution.y * resolution.z;
	const size_t n_chunk_voxels = (size_t)chunk_size.x * chunk_size.y * chunk_size.z;

	tlog::info() << "Loading " << present_chunks.size() << " zarr chunks into a " << resolution.x << "x" << resolution.y << "x"
				 << resolution.z << " volume from " << data_path;

	std::vector<uint8_t> host_data(n_voxels, 0);
	std::atomic<size_t> n_failed_chunks{0};

	ThreadPool pool;
	pool.parallel_for<size_t>(0, present_chunks.size(), [&](size_t i) {
		const ivec3 c = present_chunks[i];

		std::ifstream f{native_string(chunk_path(c)), std::ios::in | std::ios::binary | std::ios::ate};
		if (!f) {
			++n_failed_chunks;
			return;
		}

		const size_t n_bytes = (size_t)f.tellg();
		f.seekg(0);

		std::vector<uint8_t> raw(n_bytes);
		f.read((char*)raw.data(), n_bytes);

		std::vector<uint8_t> decompressed;
		const uint8_t* voxels = raw.data();
		if (is_blosc_compressed) {
			decompressed.resize(n_chunk_voxels);
			if (blosc_decompress_ctx(raw.data(), decompressed.data(), n_chunk_voxels, 1) <= 0) {
				++n_failed_chunks;
				return;
			}

			voxels = decompressed.data();
		} else if (n_bytes < n_chunk_voxels) {
			++n_failed_chunks;
			return;
		}

		const ivec3 chunk_begin = c * chunk_size;
		const ivec3 chunk_end = min(chunk_begin + chunk_size, end);
		const size_t n_x = chunk_end.x - chunk_begin.x;

		for (int z = chunk_begin.z; z < chunk_end.z; ++z) {
			for (int y = chunk_begin.y; y < chunk_end.y; ++y) {
				const size_t src = ((size_t)(z - chunk_begin.z) * chunk_size.y + (y - chunk_begin.y)) * chunk_size.x;
				const size_t dst = ((size_t)(z - begin.z) * resolution.y + (y - begin.y)) * resolution.x + (chunk_begin.x - begin.x);
				std::memcpy(host_data.data() + dst, voxels + src, n_x);
			}
		}
	});

	if (n_failed_chunks > 0) {
		throw std::runtime_error{fmt::format("Failed to read {} of {} zarr chunks.", n_failed_chunks.load(), present_chunks.size())};
	}

	m_scalar_volume.resolution = resolution;
	m_scalar_volume.data.resize_and_copy_from_host(host_data);

	m_aabb = m_render_aabb = BoundingBox{vec3(0.0f), vec3(1.0f)};
	m_render_aabb_to_local = mat3::identity();

	tlog::success() << "Loaded scalar volume of " << resolution.x << "x" << resolution.y << "x" << resolution.z << " voxels ("
					<< bytes_to_string(n_voxels) << ") after " << tlog::durationToString(std::chrono::steady_clock::now() - start);
}

__global__ void scalar_volume_slice_coords(uint32_t n_elements, uint32_t offset, int z, ivec3 resolution, vec3* __restrict__ positions) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) {
		return;
	}

	const uint32_t idx = min(i + offset, (uint32_t)resolution.x * resolution.y - 1);
	positions[i] = (vec3{(float)(idx % resolution.x), (float)(idx / resolution.x), (float)z} + 0.5f) / vec3(resolution);
}

void Testbed::save_scalar_volume_slices(const fs::path& dir) {
	if (m_testbed_mode != ETestbedMode::ScalarVolume) {
		throw std::runtime_error{"Slice dumping requires the ScalarVolume mode."};
	}

	if (!m_network) {
		throw std::runtime_error{"Slice dumping requires a network."};
	}

	const ivec3 resolution = m_scalar_volume.resolution;
	const uint32_t n_pixels = (uint32_t)resolution.x * resolution.y;
	const uint32_t max_batch_size = 1u << 20;
	const int z = resolution.z / 2;

	GPUMemory<vec3> positions(max_batch_size);
	GPUMemory<float> targets(max_batch_size);
	GPUMemory<float> predictions(max_batch_size);

	std::vector<float> targets_host(max_batch_size);
	std::vector<float> predictions_host(max_batch_size);
	std::vector<uint8_t> gt_image(n_pixels);
	std::vector<uint8_t> pred_image(n_pixels);

	double squared_error = 0.0;

	for (uint32_t offset = 0; offset < n_pixels; offset += max_batch_size) {
		const uint32_t n = std::min(max_batch_size, n_pixels - offset);
		const uint32_t batch_size = next_multiple(n, BATCH_SIZE_GRANULARITY);

		linear_kernel(scalar_volume_slice_coords, 0, nullptr, batch_size, offset, z, resolution, positions.data());
		linear_kernel(
			eval_scalar_volume_kernel, 0, nullptr, batch_size, m_scalar_volume.data.data(), positions.data(), resolution, targets.data()
		);

		GPUMatrix<float> positions_matrix((float*)positions.data(), 3, batch_size);
		GPUMatrix<float, RM> predictions_matrix(predictions.data(), 1, batch_size);
		m_network->inference(positions_matrix, predictions_matrix);

		targets.copy_to_host(targets_host);
		predictions.copy_to_host(predictions_host);

		for (uint32_t i = 0; i < n; ++i) {
			const double diff = (double)targets_host[i] - (double)predictions_host[i];
			squared_error += diff * diff;

			gt_image[offset + i] = (uint8_t)(clamp(targets_host[i], 0.0f, 1.0f) * 255.0f + 0.5f);
			pred_image[offset + i] = (uint8_t)(clamp(predictions_host[i], 0.0f, 1.0f) * 255.0f + 0.5f);
		}
	}

	const double mse = squared_error / n_pixels;
	tlog::info() << fmt::format("Slice z={} mse={:e} psnr={:.2f}dB", z, mse, -10.0 * std::log10(mse));

	write_stbi(dir / "gt_slice.png", resolution.x, resolution.y, 1, gt_image.data(), 100);
	write_stbi(dir / "pred_slice.png", resolution.x, resolution.y, 1, pred_image.data(), 100);

	tlog::success() << "Wrote slice images to " << dir;
}

} // namespace ngp
