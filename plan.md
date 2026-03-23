# Shoreline + Clouds + Water + Splashes Plan

## Goal

Build toward a final scene that renders:

- a shoreline
- volumetrically rendered clouds in the sky
- animated water
- splashes and spray hitting the shore

This repository already has a solid surface path tracer base. The missing pieces are volumetric transport, animated water, and splash/spray systems. The plan below is staged so each phase leaves the renderer usable and testable.

## Current Status

The current codebase already supports:

- progressive CUDA path tracing
- scene loading from JSON
- analytic geometry (`sphere`, `cube`)
- imported glTF meshes
- per-mesh triangle BVHs and a scene BVH
- textured materials
- normal mapping
- metallic/roughness workflow
- clearcoat and transmission
- emissive materials
- HDR environment lighting
- environment importance sampling
- direct lighting with MIS for analytic emissive geometry
- Russian roulette
- interactive scene/material/environment editing in ImGui

The current codebase does not yet support:

- volumetric media or cloud rendering
- ray marching through density fields
- a water surface system
- time-based scene animation
- particles, spray, foam, or splash simulation
- a fluid solver

## Design Principle

Do not implement clouds, water, and splashes all at once.

Recommended order:

1. Add time and animated water.
2. Make the shoreline read correctly with foam and shallow-water behavior.
3. Add volumetric clouds.
4. Add spray and splash particles.
5. Only then consider true simulation if needed.

This order gets to a convincing result much faster and fits the existing architecture.

## Phase 0: Keep the Current Renderer as the Base Platform

Use the current file ownership pattern:

- `src/sceneStructs.h`: runtime scene data definitions
- `src/scene.cpp`: JSON loading and scene setup
- `src/pathtrace.cu`: transport, intersections, and shading integration
- `src/interactions.cu`: BSDF logic
- `src/main.cpp`: editor controls and debugging UI
- `scenes/*.json`: milestone scenes and test scenes

Milestone output:

- a static shoreline scene with terrain meshes, rocks, sky, and a placeholder flat water plane

## Phase 1: Add Animated Water First

### Why

Water is the biggest visual element in the target scene. It should exist before clouds and splashes so the shoreline composition is already working.

### Files to Change

- `src/sceneStructs.h`
- `src/scene.cpp`
- `src/pathtrace.cu`
- `src/main.cpp`

### Files to Add

- `src/water.h`
- `src/water.cu`

### What to Add

In `src/sceneStructs.h`:

- add a `WaterSettings` struct
- add scene-level time data to `RenderState`

Suggested `WaterSettings` fields:

- enabled flag
- water plane height or transform
- amplitude
- wavelength
- speed
- steepness
- normal strength
- water color / absorption tint
- IOR
- foam amount

In `src/scene.cpp`:

- extend scene JSON parsing with a `"Water"` block
- load water settings into `RenderState`

In `src/water.cu`:

- implement a procedural wave surface evaluator
- start with Gerstner waves or a procedural heightfield
- implement a normal evaluator for the displaced surface

In `src/pathtrace.cu`:

- add a dedicated water intersection path
- add water shading using the existing reflection/transmission framework
- use a single large procedural water surface instead of simulated geometry for the first version

In `src/main.cpp`:

- add UI controls for water amplitude, speed, tint, and roughness-related parameters

### Milestone Output

- animated ocean water with reflections and refraction
- shoreline composition with usable camera framing

## Phase 2: Make the Shoreline Believable

### Why

Before clouds or splashes, the water needs to read as shallow near shore and active where waves break.

### Files to Change

- `src/pathtrace.cu`
- `src/interactions.cu`
- `src/scene.cpp`
- `src/sceneStructs.h`

### What to Add

In `src/pathtrace.cu`:

- foam heuristics from wave steepness
- foam near shore from shallow depth or shoreline mask
- optional whitewater contribution on crest regions

In `src/interactions.cu`:

- tune dielectric water behavior around `IOR = 1.333`
- keep water strongly Fresnel-driven
- add simple Beer-Lambert style attenuation for refracted paths if practical

In `src/sceneStructs.h` and `src/scene.cpp`:

- add shoreline/foam tuning controls if needed

### Milestone Output

- shallow water near shore
- darker offshore water
- visible foam bands where waves break

## Phase 3: Add Volumetric Clouds

### Why

Clouds require a separate transport path. They should be built after the water scene is already visually grounded.

### Files to Change

- `src/sceneStructs.h`
- `src/scene.cpp`
- `src/pathtrace.cu`
- `src/main.cpp`

### Files to Add

- `src/volume.h`
- `src/volume.cu`

### What to Add

In `src/sceneStructs.h`:

- add `CloudVolume` or `VolumeSettings`
- represent a bounded volume with a world-space box

Suggested volume fields:

- enabled flag
- bounding box min/max
- density scale
- noise scale
- extinction
- scattering albedo
- phase anisotropy `g`
- march step size

In `src/scene.cpp`:

- extend JSON parsing with a `"Volumes"` block

In `src/volume.cu`:

- add density evaluation
- add procedural noise sampling
- add ray-box entry/exit calculation
- add volume ray marching helpers

In `src/pathtrace.cu`:

- march through cloud volumes before the nearest surface hit
- implement single scattering first
- support sun/environment lighting contribution through the volume

In `src/main.cpp`:

- expose cloud density, step size, and anisotropy controls

### Important Constraint

Do single scattering first. Do not start with full multiple-scattering clouds.

### Milestone Output

- shoreline scene with a believable cloud bank or cloud layer
- visible depth and lighting variation in the clouds

## Phase 4: Add Spray and Splashes as Particles

### Why

You do not need a full fluid solver to get convincing shoreline splashes. Spray particles get you much closer, much faster.

### Files to Change

- `src/sceneStructs.h`
- `src/scene.cpp`
- `src/pathtrace.cu`
- `src/main.cpp`

### Files to Add

- `src/particles.h`
- `src/particles.cu`

### What to Add

In `src/sceneStructs.h`:

- add particle or spray emitter data

In `src/scene.cpp`:

- parse authored splash emitters or shoreline spray zones from JSON

In `src/particles.cu`:

- implement particle spawn/update logic
- start with simple ballistic motion: velocity, gravity, lifetime

In `src/pathtrace.cu`:

- render droplets as small spheres or another simple primitive
- integrate them using the existing intersection and shading framework

In `src/main.cpp`:

- add controls for spawn rate, droplet size, lifetime, and spray intensity

### Milestone Output

- visible shore spray
- splashes near rocks or wave impact zones

## Phase 5: Only Then Consider True Simulation

### Why

A true fluid solver is expensive and is not the fastest route to your target image quality.

### Files to Add

- `src/simulation.h`
- `src/simulation.cu`

### What to Add

- per-frame simulation update loop
- optional water-state cache or particle solver
- only after the procedural and particle-based version is already working

### Recommendation

Do not begin with FLIP or SPH. Only move here if the procedural water + foam + spray approach is not enough for your final result.

## Cross-Cutting Change: Add Time to the Scene

This should happen early because water and particles need it.

### Files to Change

- `src/sceneStructs.h`
- `src/main.cpp`
- `src/pathtrace.cu`

### What to Add

- `RenderState.time`
- a frame delta or accumulated animation time
- a UI toggle for pausing or advancing animation

Without scene time, water animation and spray playback will be awkward.

## Suggested Scene Files

Add milestone scene files such as:

- `scenes/shoreline_base.json`
- `scenes/shoreline_water.json`
- `scenes/shoreline_foam.json`
- `scenes/shoreline_clouds.json`
- `scenes/shoreline_spray.json`

Each should isolate one feature milestone so debugging stays manageable.

## Recommended Implementation Order

1. Add `RenderState.time`.
2. Add `WaterSettings` and a procedural water surface.
3. Add water shading and shoreline foam.
4. Build a dedicated shoreline test scene.
5. Add `CloudVolume` and single-scattering clouds.
6. Add spray particles and simple splash emitters.
7. Reassess whether true simulation is still necessary.

## Practical Recommendation for This Codebase

The fastest path to a convincing final result in this repository is:

- procedural animated water
- shoreline foam
- volumetric cloud box or cloud layer
- spray particles

That route matches the current architecture much better than jumping immediately into full fluid simulation.
