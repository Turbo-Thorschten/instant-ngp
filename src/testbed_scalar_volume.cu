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
 *  @brief  Fits a multi-resolution dense uint8 intensity volume, f(x in [0,1]^3, s in [0,1]) -> intensity.
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

static constexpr uint32_t MAX_SCALAR_VOLUME_LEVELS = 8;

// All levels are addressed in the normalized [0,1]^3 of level 0: a position x is mapped to
// level-l voxel coordinates via x * scale + offset, with
// scale = resolution_0 / 2^l and offset = begin_0 / 2^l - begin_l.
struct ScalarVolumeLevels {
	const uint8_t* data[MAX_SCALAR_VOLUME_LEVELS];
	ivec3 resolution[MAX_SCALAR_VOLUME_LEVELS];
	vec3 scale[MAX_SCALAR_VOLUME_LEVELS];
	vec3 offset[MAX_SCALAR_VOLUME_LEVELS];
	uint32_t n_levels;
};

struct ScalarVolumeLevelCdf {
	float cdf[MAX_SCALAR_VOLUME_LEVELS];
};

inline NGP_HOST_DEVICE float sample_scalar_volume(const ScalarVolumeLevels& levels, uint32_t level, const vec3& x) {
	const ivec3 resolution = levels.resolution[level];
	const uint8_t* __restrict__ data = levels.data[level];

	auto read_val = [&](int px, int py, int pz) {
		return (float)data[((size_t)pz * resolution.y + py) * resolution.x + px] * (1.0f / 255.0f);
	};

	vec3 pos = clamp(x * levels.scale[level] + levels.offset[level] - 0.5f, 0.0f, vec3(resolution) - (1.0f + 1e-4f));

	const ivec3 pos_int = pos;
	const vec3 weight = pos - vec3(pos_int);

	const ivec3 idx = clamp(pos_int, 0, resolution - 2);

	return (1 - weight.x) * (1 - weight.y) * (1 - weight.z) * read_val(idx.x, idx.y, idx.z) +
		(weight.x) * (1 - weight.y) * (1 - weight.z) * read_val(idx.x + 1, idx.y, idx.z) +
		(1 - weight.x) * (weight.y) * (1 - weight.z) * read_val(idx.x, idx.y + 1, idx.z) +
		(weight.x) * (weight.y) * (1 - weight.z) * read_val(idx.x + 1, idx.y + 1, idx.z) +
		(1 - weight.x) * (1 - weight.y) * (weight.z) * read_val(idx.x, idx.y, idx.z + 1) +
		(weight.x) * (1 - weight.y) * (weight.z) * read_val(idx.x + 1, idx.y, idx.z + 1) +
		(1 - weight.x) * (weight.y) * (weight.z) * read_val(idx.x, idx.y + 1, idx.z + 1) +
		(weight.x) * (weight.y) * (weight.z) * read_val(idx.x + 1, idx.y + 1, idx.z + 1);
}

__global__ void eval_scalar_volume_kernel(
	uint32_t n_elements, ScalarVolumeLevels levels, uint32_t level, const vec4* __restrict__ positions, float* __restrict__ result
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) {
		return;
	}

	result[i] = sample_scalar_volume(levels, level, vec3(positions[i]));
}

// Consumes the 4th uniform random number of each sample to draw its level and replaces it with the level's scale s.
// s = level / n_levels rather than level / (n_levels - 1) because tcnn's OneBlob encoding wraps around, which would
// make s = 1 indistinguishable from s = 0.
__global__ void sample_scalar_volume_kernel(
	uint32_t n_elements,
	ScalarVolumeLevels levels,
	ScalarVolumeLevelCdf level_cdf,
	float inv_n_levels,
	vec4* __restrict__ positions,
	float* __restrict__ targets
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) {
		return;
	}

	const vec4 pos = positions[i];

	uint32_t level = 0;
	while (level + 1 < levels.n_levels && pos.w >= level_cdf.cdf[level]) {
		++level;
	}

	targets[i] = sample_scalar_volume(levels, level, vec3(pos));
	positions[i].w = (float)level * inv_n_levels;
}

Testbed::NetworkDims Testbed::network_dims_scalar_volume() const {
	NetworkDims dims;
	dims.n_input = 4;
	dims.n_output = 1;
	dims.n_pos = 3;
	return dims;
}

static ScalarVolumeLevels scalar_volume_levels(const Testbed::ScalarVolume& volume) {
	ScalarVolumeLevels result;
	result.n_levels = (uint32_t)volume.levels.size();

	for (uint32_t l = 0; l < result.n_levels; ++l) {
		const auto& level = volume.levels[l];
		const float inv_level_scale = 1.0f / (float)(1u << l);

		result.data[l] = level.data;
		result.resolution[l] = level.resolution;
		result.scale[l] = vec3(volume.resolution) * inv_level_scale;
		result.offset[l] = vec3(volume.begin) * inv_level_scale - vec3(level.begin);
	}

	return result;
}

// Weight of each unlocked level, capped such that the coarse levels keep a noticeable share of the batch.
static ScalarVolumeLevelCdf scalar_volume_level_cdf(uint32_t n_levels, uint32_t n_unlocked_levels) {
	ScalarVolumeLevelCdf result;

	float total = 0.0f;
	for (uint32_t l = 0; l < n_levels; ++l) {
		const float weight = l + n_unlocked_levels < n_levels ? 0.0f : std::min(std::pow(8.0f, (float)(n_levels - 1 - l)), 64.0f);
		total += weight;
		result.cdf[l] = total;
	}

	for (uint32_t l = 0; l < n_levels; ++l) {
		result.cdf[l] /= total;
	}

	return result;
}

void Testbed::train_scalar_volume(size_t target_batch_size, bool get_loss_scalar, cudaStream_t stream) {
	const uint32_t n_output_dims = 1;
	const uint32_t n_input_dims = 4;

	const uint32_t batch_size = (uint32_t)target_batch_size;
	const uint32_t n_levels = (uint32_t)m_scalar_volume.levels.size();

	uint32_t n_unlocked_levels = n_levels;
	if (m_scalar_volume_curriculum_steps > 0) {
		n_unlocked_levels = std::min(n_levels, 1 + (uint32_t)(m_training_step / m_scalar_volume_curriculum_steps));
	}

	if (n_unlocked_levels != m_scalar_volume.n_unlocked_levels) {
		m_scalar_volume.n_unlocked_levels = n_unlocked_levels;
		tlog::info() << "step=" << m_training_step << " curriculum unlocked levels " << (n_levels - n_unlocked_levels) << ".."
					 << (n_levels - 1);
	}

	m_scalar_volume.training.positions.enlarge(batch_size);
	m_scalar_volume.training.targets.enlarge(batch_size);

	generate_random_uniform<float>(stream, m_rng, batch_size * n_input_dims, (float*)m_scalar_volume.training.positions.data());

	linear_kernel(
		sample_scalar_volume_kernel,
		0,
		stream,
		batch_size,
		scalar_volume_levels(m_scalar_volume),
		scalar_volume_level_cdf(n_levels, n_unlocked_levels),
		1.0f / (float)n_levels,
		m_scalar_volume.training.positions.data(),
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
	vec4* __restrict__ positions,
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
		positions[idx] = vec4(-1.0f, -1.0f, -1.0f, 0.0f);
		return;
	}

	// plane_z is negative, so this lands on the slicing plane in front of the camera.
	const vec3 pos = ray.o - plane_z * ray.d;
	positions[idx] = vec4(pos.x, pos.y, pos.z, 0.0f);
	depth_buffer[idx] = -plane_z;
}

__global__ void shade_kernel_scalar_volume(
	ivec2 resolution, const vec4* __restrict__ positions, const float* __restrict__ values, vec4* __restrict__ frame_buffer
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	uint32_t idx = x + resolution.x * y;

	const vec3 pos = vec3(positions[idx]);
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
			scalar_volume_levels(m_scalar_volume),
			0,
			m_scalar_volume.render_coords.data(),
			m_scalar_volume.render_out.data()
		);
	} else {
		GPUMatrix<float> positions_matrix((float*)m_scalar_volume.render_coords.data(), 4, n_elements);
		GPUMatrix<float, RM> values_matrix(m_scalar_volume.render_out.data(), 1, n_elements);
		m_network->inference(stream, positions_matrix, values_matrix);
	}

	shade_kernel_scalar_volume<<<blocks, threads, 0, stream>>>(
		res, m_scalar_volume.render_coords.data(), m_scalar_volume.render_out.data(), render_buffer.frame_buffer
	);
}

struct ZarrLevelInfo {
	ivec3 resolution = ivec3(0);
	ivec3 begin = ivec3(0);
	ivec3 end = ivec3(0);
	ivec3 chunk_size = ivec3(0);
	std::vector<ivec3> present_chunks;
	bool is_blosc_compressed = false;
	std::string separator = ".";
};

static fs::path zarr_chunk_path(const fs::path& level_path, const std::string& separator, const ivec3& c) {
	if (separator == "/") {
		return level_path / std::to_string(c.z) / std::to_string(c.y) / std::to_string(c.x);
	}

	return level_path / fmt::format("{}{}{}{}{}", c.z, separator, c.y, separator, c.x);
}

static ZarrLevelInfo scan_zarr_level(const fs::path& level_path) {
	std::ifstream metadata_file{native_string(level_path / ".zarray")};
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

	ZarrLevelInfo info;

	if (metadata.contains("compressor") && !metadata["compressor"].is_null()) {
		const std::string compressor_id = metadata["compressor"].value("id", std::string{});
		if (compressor_id != "blosc") {
			throw std::runtime_error{fmt::format("Unsupported zarr compressor '{}'.", compressor_id)};
		}

		info.is_blosc_compressed = true;
	}

	info.separator = metadata.value("dimension_separator", std::string{"."});
	const bool is_nested = info.separator == "/";

	// zarr axis order is z, y, x; everything below is in x, y, z order.
	const ivec3 volume_shape = {shape[2], shape[1], shape[0]};
	info.chunk_size = {chunks[2], chunks[1], chunks[0]};
	const ivec3 n_chunks = {
		div_round_up(volume_shape.x, info.chunk_size.x),
		div_round_up(volume_shape.y, info.chunk_size.y),
		div_round_up(volume_shape.z, info.chunk_size.z)
	};

	for (int cz = 0; cz < n_chunks.z; ++cz) {
		if (is_nested && !(level_path / std::to_string(cz)).exists()) {
			continue;
		}

		for (int cy = 0; cy < n_chunks.y; ++cy) {
			if (is_nested && !(level_path / std::to_string(cz) / std::to_string(cy)).exists()) {
				continue;
			}

			for (int cx = 0; cx < n_chunks.x; ++cx) {
				if (zarr_chunk_path(level_path, info.separator, {cx, cy, cz}).exists()) {
					info.present_chunks.emplace_back(cx, cy, cz);
				}
			}
		}
	}

	if (info.present_chunks.empty()) {
		throw std::runtime_error{fmt::format("No zarr chunks found in '{}'.", level_path.str())};
	}

	ivec3 chunk_min = info.present_chunks.front();
	ivec3 chunk_max = info.present_chunks.front();
	for (const auto& c : info.present_chunks) {
		chunk_min = min(chunk_min, c);
		chunk_max = max(chunk_max, c);
	}

	info.begin = chunk_min * info.chunk_size;
	info.end = min((chunk_max + 1) * info.chunk_size, volume_shape);
	info.resolution = info.end - info.begin;

	return info;
}

static void read_zarr_level(const fs::path& level_path, const ZarrLevelInfo& info, uint8_t* dst) {
	const size_t n_voxels = (size_t)info.resolution.x * info.resolution.y * info.resolution.z;
	const size_t n_chunk_voxels = (size_t)info.chunk_size.x * info.chunk_size.y * info.chunk_size.z;

	std::memset(dst, 0, n_voxels);

	std::atomic<size_t> n_failed_chunks{0};

	ThreadPool pool;
	pool.parallel_for<size_t>(0, info.present_chunks.size(), [&](size_t i) {
		const ivec3 c = info.present_chunks[i];

		std::ifstream f{native_string(zarr_chunk_path(level_path, info.separator, c)), std::ios::in | std::ios::binary | std::ios::ate};
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
		if (info.is_blosc_compressed) {
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

		const ivec3 chunk_begin = c * info.chunk_size;
		const ivec3 chunk_end = min(chunk_begin + info.chunk_size, info.end);
		const size_t n_x = chunk_end.x - chunk_begin.x;

		for (int z = chunk_begin.z; z < chunk_end.z; ++z) {
			for (int y = chunk_begin.y; y < chunk_end.y; ++y) {
				const size_t src = ((size_t)(z - chunk_begin.z) * info.chunk_size.y + (y - chunk_begin.y)) * info.chunk_size.x;
				const size_t dstoff = ((size_t)(z - info.begin.z) * info.resolution.y + (y - info.begin.y)) * info.resolution.x +
					(chunk_begin.x - info.begin.x);
				std::memcpy(dst + dstoff, voxels + src, n_x);
			}
		}
	});

	if (n_failed_chunks > 0) {
		throw std::runtime_error{fmt::format("Failed to read {} of {} zarr chunks.", n_failed_chunks.load(), info.present_chunks.size())};
	}
}

void Testbed::load_scalar_volume(const fs::path& data_path) {
	auto start = std::chrono::steady_clock::now();

	std::vector<ZarrLevelInfo> infos;
	for (uint32_t l = 0; l < MAX_SCALAR_VOLUME_LEVELS; ++l) {
		const fs::path level_path = data_path / std::to_string(l);
		if (!(level_path / ".zarray").exists()) {
			break;
		}

		infos.emplace_back(scan_zarr_level(level_path));
	}

	if (infos.empty()) {
		throw std::runtime_error{fmt::format("Zarr metadata '{}' does not exist.", (data_path / "0" / ".zarray").str())};
	}

	size_t total_bytes = 0;
	for (const auto& info : infos) {
		total_bytes += (size_t)info.resolution.x * info.resolution.y * info.resolution.z;
	}

	m_scalar_volume.levels.clear();
	m_scalar_volume.levels.resize(infos.size());

	for (size_t l = 0; l < infos.size(); ++l) {
		const auto& info = infos[l];
		auto& level = m_scalar_volume.levels[l];

		const fs::path level_path = data_path / std::to_string(l);
		const size_t n_voxels = (size_t)info.resolution.x * info.resolution.y * info.resolution.z;

		tlog::info() << "Loading level " << l << ": " << info.present_chunks.size() << " zarr chunks into a " << info.resolution.x << "x"
					 << info.resolution.y << "x" << info.resolution.z << " volume at offset " << info.begin.x << "," << info.begin.y << ","
					 << info.begin.z;

		level.resolution = info.resolution;
		level.begin = info.begin;

		std::vector<uint8_t> host_data(n_voxels);
		read_zarr_level(level_path, info, host_data.data());
		level.vram.resize_and_copy_from_host(host_data);
		level.data = level.vram.data();
	}

	m_scalar_volume.resolution = m_scalar_volume.levels.front().resolution;
	m_scalar_volume.begin = m_scalar_volume.levels.front().begin;
	m_scalar_volume.n_unlocked_levels = 0;

	m_aabb = m_render_aabb = BoundingBox{vec3(0.0f), vec3(1.0f)};
	m_render_aabb_to_local = mat3::identity();

	tlog::success() << "Loaded " << infos.size() << " scalar volume levels (" << bytes_to_string(total_bytes) << ") after "
					<< tlog::durationToString(std::chrono::steady_clock::now() - start);
}

__global__ void scalar_volume_slice_coords(
	uint32_t n_elements, uint32_t offset, int z, ivec3 resolution, float s, vec4* __restrict__ positions
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) {
		return;
	}

	const uint32_t idx = min(i + offset, (uint32_t)resolution.x * resolution.y - 1);
	const vec3 pos = (vec3{(float)(idx % resolution.x), (float)(idx / resolution.x), (float)z} + 0.5f) / vec3(resolution);
	positions[i] = vec4(pos.x, pos.y, pos.z, s);
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

	const uint32_t n_levels = (uint32_t)m_scalar_volume.levels.size();
	const float inv_n_levels = 1.0f / (float)n_levels;
	const auto levels = scalar_volume_levels(m_scalar_volume);

	GPUMemory<vec4> positions(max_batch_size);
	GPUMemory<float> targets(max_batch_size);
	GPUMemory<float> predictions(max_batch_size);

	std::vector<float> targets_host(max_batch_size);
	std::vector<float> predictions_host(max_batch_size);
	std::vector<uint8_t> gt_image(n_pixels);
	std::vector<uint8_t> pred_image(n_pixels);
	std::vector<float> level0_gt(n_pixels);
	std::vector<float> prev_gt(n_pixels);
	std::vector<float> cur_gt(n_pixels);

	for (uint32_t level = 0; level < n_levels; ++level) {
		const float s = (float)level * inv_n_levels;

		double squared_error = 0.0;
		double sum_gt = 0.0, sum_gt2 = 0.0;
		double sum_ref = 0.0, sum_ref2 = 0.0, sum_gt_ref = 0.0;
		double sum_prev = 0.0, sum_prev2 = 0.0, sum_gt_prev = 0.0;

		for (uint32_t offset = 0; offset < n_pixels; offset += max_batch_size) {
			const uint32_t n = std::min(max_batch_size, n_pixels - offset);
			const uint32_t batch_size = next_multiple(n, BATCH_SIZE_GRANULARITY);

			linear_kernel(scalar_volume_slice_coords, 0, nullptr, batch_size, offset, z, resolution, s, positions.data());
			linear_kernel(eval_scalar_volume_kernel, 0, nullptr, batch_size, levels, level, positions.data(), targets.data());

			GPUMatrix<float> positions_matrix((float*)positions.data(), 4, batch_size);
			GPUMatrix<float, RM> predictions_matrix(predictions.data(), 1, batch_size);
			m_network->inference(positions_matrix, predictions_matrix);

			targets.copy_to_host(targets_host);
			predictions.copy_to_host(predictions_host);

			for (uint32_t i = 0; i < n; ++i) {
				const double diff = (double)targets_host[i] - (double)predictions_host[i];
				squared_error += diff * diff;

				const double gt = targets_host[i];
				const double ref = level == 0 ? gt : level0_gt[offset + i];
				const double prev = level == 0 ? gt : prev_gt[offset + i];
				sum_gt += gt;
				sum_gt2 += gt * gt;
				sum_ref += ref;
				sum_ref2 += ref * ref;
				sum_gt_ref += gt * ref;
				sum_prev += prev;
				sum_prev2 += prev * prev;
				sum_gt_prev += gt * prev;

				cur_gt[offset + i] = (float)gt;

				if (level == 0) {
					level0_gt[offset + i] = (float)gt;
					gt_image[offset + i] = (uint8_t)(clamp(targets_host[i], 0.0f, 1.0f) * 255.0f + 0.5f);
				}

				pred_image[offset + i] = (uint8_t)(clamp(predictions_host[i], 0.0f, 1.0f) * 255.0f + 0.5f);
			}
		}

		const double mse = squared_error / n_pixels;
		auto correlation = [n = (double)n_pixels](double sum_a, double sum_b, double sum_a2, double sum_b2, double sum_ab) {
			const double cov = sum_ab / n - (sum_a / n) * (sum_b / n);
			const double var_a = sum_a2 / n - (sum_a / n) * (sum_a / n);
			const double var_b = sum_b2 / n - (sum_b / n) * (sum_b / n);
			return cov / std::sqrt(var_a * var_b);
		};

		tlog::info() << fmt::format(
			"Slice z={} level={} s={:.3f} mse={:e} psnr={:.2f}dB | gt_mean={:.4f} gt_std={:.4f} corr_vs_level0={:.4f} corr_vs_prev={:.4f}",
			z,
			level,
			s,
			mse,
			-10.0 * std::log10(mse),
			sum_gt / n_pixels,
			std::sqrt(sum_gt2 / n_pixels - (sum_gt / n_pixels) * (sum_gt / n_pixels)),
			correlation(sum_gt, sum_ref, sum_gt2, sum_ref2, sum_gt_ref),
			correlation(sum_gt, sum_prev, sum_gt2, sum_prev2, sum_gt_prev)
		);

		prev_gt.swap(cur_gt);

		if (level == 0) {
			write_stbi(dir / "gt_slice.png", resolution.x, resolution.y, 1, gt_image.data(), 100);
			write_stbi(dir / "pred_slice.png", resolution.x, resolution.y, 1, pred_image.data(), 100);
		} else if (level == 3) {
			write_stbi(dir / "pred_slice_s3.png", resolution.x, resolution.y, 1, pred_image.data(), 100);
		}
	}

	tlog::success() << "Wrote slice images to " << dir;
}

} // namespace ngp
