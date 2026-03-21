#pragma once

// Small general-purpose epsilon used for floating-point comparisons.
#define RENDER_CONFIG_EPSILON 1.0e-5f

// Default minimum ray hit distance for most analytic intersections.
#define RENDER_CONFIG_MIN_INTERSECTION_T 1.0e-4f

// Base origin bias used to offset spawned rays away from the surface.
#define RENDER_CONFIG_RAY_ORIGIN_BIAS 1.0e-3f

// Scene-BVH traversal stack size on the GPU.
// Increased during importer debugging to prevent missed hits from stack overflow.
#define RENDER_CONFIG_MAX_SCENE_BVH_STACK_SIZE 64

// Per-mesh triangle-BVH traversal stack size on the GPU.
// Kept large for deep imported meshes and irregular triangle hierarchies.
#define RENDER_CONFIG_MAX_TRIANGLE_BVH_STACK_SIZE 64

// CPU-side BVH traversal stack size used for object picking and debug traversal.
#define RENDER_CONFIG_MAX_PICKING_BVH_STACK_SIZE 1024

// Triangle determinant epsilon used by Moller-Trumbore intersection.
// Smaller than the global epsilon so tiny imported triangles remain hittable.
#define RENDER_CONFIG_TRIANGLE_DET_EPSILON 1.0e-8f

// Triangle-specific minimum hit distance.
// Smaller than the global default to avoid dropping hits on very small triangles.
#define RENDER_CONFIG_TRIANGLE_MIN_INTERSECTION_T 1.0e-6f

// UV determinant epsilon used when deriving tangent frames from triangle UVs.
#define RENDER_CONFIG_TRIANGLE_UV_DET_EPSILON 1.0e-8f

// Maximum number of transparent/transmissive surfaces a shadow ray will march through.
#define RENDER_CONFIG_SHADOW_TRANSMITTANCE_MAX_STEPS 32

// Extra direction-based ray offset scale for thin masked/transmissive geometry.
// This helps avoid immediate self-rehits on foliage cards and thin glass sheets.
#define RENDER_CONFIG_THIN_SURFACE_DIRECTION_OFFSET_SCALE 8.0f

// Bounce index where Russian roulette begins.
#define RENDER_CONFIG_RUSSIAN_ROULETTE_START_BOUNCE 4

// Lower bound on Russian roulette survival probability.
#define RENDER_CONFIG_MIN_RUSSIAN_ROULETTE_SURVIVAL 0.1f

// Upper bound on Russian roulette survival probability.
#define RENDER_CONFIG_MAX_RUSSIAN_ROULETTE_SURVIVAL 0.95f

// 1D CUDA block size used by tracing, sorting, and shading kernels.
#define RENDER_CONFIG_PATH_TRACE_BLOCK_SIZE_1D 256

// 2D CUDA block size used for camera ray generation and final image upload.
#define RENDER_CONFIG_CAMERA_BLOCK_SIZE_X 8
#define RENDER_CONFIG_CAMERA_BLOCK_SIZE_Y 8

// Default material-sort cadence when sorting is enabled from the UI.
#define RENDER_CONFIG_DEFAULT_SORT_EVERY_N_ITERATIONS 4

// Default maximum bounce that participates in material sorting.
#define RENDER_CONFIG_DEFAULT_SORT_MAX_BOUNCE 2

// Default minimum number of active paths before material sorting is worth the overhead.
#define RENDER_CONFIG_DEFAULT_SORT_MIN_PATH_COUNT 32768
