# Spatial Memory

Task Memory knows *what* the route asks for and Temporal Memory judges the
last few frames; neither remembers *where* anything is. Spatial Memory is the
agent's map: a floor-plane occupancy grid fused from every RGB-D frame, the
Captioner's landmarks anchored in world coordinates, and the one world point
the agent is currently walking to. It never emits actions.

```
spatial_memory/
  occupancy_grid.py   OccupancyGrid: depth -> FREE/OCCUPIED/UNKNOWN cells,
                      frontiers, A* planner over free (and, at a cost,
                      unknown) cells
  landmarks.py        LandmarkRegistry: repeated sightings merge into a
                      running-mean world point per landmark name
  targets.py          CommittedTarget: reached / passed / stagnant / stale
                      lifecycle of the point the agent committed to
  spatial_memory.py   SpatialMemory: the facade the agent talks to
```

## Per-step protocol

```python
spatial.observe(step=t, depth_m=depth, intrinsics=K, camera_to_world=T,
                floor_mask=actor_floor_mask)          # fuse the frame
spatial.register_landmark(name, world_xyz, subgoal_id=sg)  # when the Captioner
                                                           # localised one
target = spatial.active_target(current_subgoal_id)   # None once reached/stale
if target is None:
    ...ask the waypoint model, then...
    spatial.commit_target(point.world_xyz, kind="model_waypoint", subgoal_id=sg)
waypoint, remaining_m, how = spatial.next_waypoint()  # lookahead on the A* path
```

The grid frame follows Habitat: `y` up, camera forward is `-z`, floor plane
is `(x, z)`. Cells are 0.10 m; the map is 60 m square, centred on the first
pose. A cell is FREE when the actor's floor mask (or, without a camera
height, a height band around the floor) hits it, OCCUPIED when enough depth
returns land 0.3-1.8 m above the floor, and the agent's own footprint is
always free.

## Why a committed target

The doorway lock in `vln_agent_4` already proved the pattern: once a world
point is known, walking to it without re-querying the VLM removes the
back-and-forth that a fresh pixel choice every 0.25 m produces, and cuts the
model calls per episode. Spatial Memory generalises the lock to every stage
that walks somewhere (model waypoint, located landmark, preview heading,
frontier), with a shared lifecycle: reached within tolerance, walked past,
no closer for N steps (stagnant), or older than the age budget (stale).

## Frontiers

FREE cells with an UNKNOWN 4-neighbour, clustered 8-connected. `choose_frontier`
scores clusters by distance to an ideal range, alignment with a preferred
bearing (the heading the model or the subgoal suggests), and size; boundaries
the agent already walked to are skipped so it does not oscillate between two
openings.
