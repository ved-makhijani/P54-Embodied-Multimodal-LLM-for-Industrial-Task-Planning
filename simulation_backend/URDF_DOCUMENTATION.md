# URDF Asset Documentation

**Project:** P54 — Embodied Multimodal LLM for Industrial Task Planning  
**Package:** `simulation_backend/assets/urdf/`  
**Last updated:** May 2026

---

## Overview

This document covers all URDF (Unified Robot Description Format) files used in the simulation backend of this project. It details the origin, authorship, licence status, physical properties, and intended usage of each file.

URDF is an XML-based format used by ROS and PyBullet to describe the physical, visual, and collision properties of robots and objects in a simulation environment. Each URDF defines one or more **links** (rigid bodies) and optionally **joints** (connections between links).

---

## Authorship and Licence

### Custom-authored URDFs (this project)

The following URDF files were authored for this project by the P54 development team :

| File | Author | Date |
|---|---|---|
| `block.urdf` | P54 Team | May 2026 |
| `tray.urdf` | P54 Team | May 2026 |
| `table.urdf` | P54 Team | May 2026 |
| `workstation.urdf` | P54 Team | May 2026 |

These files are original works created for the COS40005 Computing Technology Project A at Swinburne University of Technology in collaboration with ARENA2036. They are not derived from, copied from, or adapted from any third-party URDF repositories.

**Licence:** These files are released under the same licence as the broader P54 project repository. If no explicit project licence is specified, all rights are reserved by the authors and Swinburne University of Technology.

---

### Third-party URDFs (robot models — not in this folder)

Robot URDFs used in `simulation_backend/robots/` are sourced from third parties and carry their own licences. These are documented here for completeness.

**Franka Panda (`franka_panda/panda.urdf`)**  
Source: Bullet3 data repository / Franka Robotics GmbH  
Repository: https://github.com/bulletphysics/bullet3/tree/master/data/franka_panda  
Licence: The Franka Robotics URDF is provided for research and educational use. See the Franka Robotics terms of use at https://www.franka.de. The version bundled with PyBullet is distributed under the terms of the Bullet3 software licence (zlib licence).

**Kuka IIWA (`kuka_iiwa/model.urdf`)**  
Source: Bullet3 / pybullet_data package  
Repository: https://github.com/bulletphysics/bullet3/tree/master/data/kuka_iiwa  
Licence: Distributed as part of pybullet_data under the zlib/libpng licence. The Kuka IIWA robot model is provided for simulation and research purposes.

**Universal Robots UR5**  
Source: Universal Robots ROS2 Description / community PyBullet port  
Repository: https://github.com/UniversalRobots/Universal_Robots_ROS2_Description  
Licence: Apache License 2.0. See https://www.apache.org/licenses/LICENSE-2.0

**Robotiq 85 Gripper**  
Source: ROS Industrial / robotiq package  
Repository: https://github.com/ros-industrial/robotiq  
Licence: Apache License 2.0.

> **Note:** Robot URDF files should be reviewed against their respective upstream licences before any commercial use or redistribution.

**Usage in pipeline:**
```yaml
# scene_config.yaml
- label: <label_to_display_in_Pybullet>
  urdf:  simulation_backend/assets/urdf/<filename>.urdf
  color: [0.35, 0.35, 0.35, 1.0] 
  position: [0.80, 0.00, 0.0]
  mass_kg: 0.0
  graspable: false
```

---

## URDF Format Reference

All files in this package conform to the URDF specification maintained by the ROS project. The authoritative specification is available at:

> Quigley, M., Conley, K., Gerkey, B., Faust, J., Foote, T., Leibs, J., Wheeler, R., & Ng, A. Y. (2009). ROS: an open-source Robot Operating System. *ICRA Workshop on Open Source Software*, 3(3.2), 5. https://www.ros.org

URDF format documentation: https://wiki.ros.org/urdf/XML

PyBullet URDF loading documentation: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA

---

## Modifying or Adding URDFs

To add a new object to the simulation:

1. Create a new `.urdf` file in `simulation_backend/assets/urdf/` using primitive geometry (`<box>`, `<sphere>`, `<cylinder>`)
2. Add a new entry to the `objects` list in `simulation_backend/scene_config.yaml`
3. No code changes are required — `object_loader.py` reads the config and loads all entries automatically

To modify an existing object's dimensions, edit the `size` attribute in both the `<visual>` and `<collision>` elements. Always keep both consistent — mismatched visual and collision geometry causes objects to appear in different positions than where physics interactions occur.

To change an object's default colour, edit the `<color rgba>` in the URDF or update the `color` field in `scene_config.yaml`. The `scene_config.yaml` value takes precedence at runtime since `object_loader.py` calls `changeVisualShape()` after loading.

---

## Coordinate System

All URDF files use the standard ROS/PyBullet right-handed coordinate system:

- **X axis** — forward (away from robot base by convention)
- **Y axis** — left
- **Z axis** — up

Distances are in **metres**. Angles are in **radians**. Mass is in **kilograms**.
