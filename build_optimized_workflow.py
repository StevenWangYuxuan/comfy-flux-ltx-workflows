#!/usr/bin/env python3
"""Build optimized FLUX+LTX I2V workflow with STG, better params, and car scene prompts."""

import json
import copy

# Load original
with open("/home/steven/comfy/flux_ltx_i2v.json") as f:
    wf = json.load(f)

# === 1. Update FLUX prompts for car scene ===
for node in wf["nodes"]:
    if node["id"] == 5:  # CLIPTextEncodeFlux
        node["widgets_values"] = [
            # clip_l: short keywords
            "sleek matte black hypercar parked on wet asphalt road, golden sunset backlight, cinematic automotive photography, low angle three-quarter front hero shot, dramatic rim lighting on car body curves, wet pavement reflections",
            # t5xxl: detailed scene
            "A breathtaking cinematic wide shot of a matte black hypercar parked diagonally across a rain-dampened asphalt road during golden hour. The car is perfectly static, engine off, captured from a low three-quarter front angle that emphasizes its aggressive aerodynamic stance and sculpted body curves. The setting sun behind the car creates a golden-orange halo tracing the vehicle's silhouette, while the wet pavement mirrors the sky in deep amber and crimson reflections. Thin wisps of heat haze rise from the still-warm carbon ceramic brake discs. The background reveals a winding coastal road with dramatic cliffs and ocean on one side, palm trees silhouetted against the sunset sky. No driver or people visible. The atmosphere is tense and anticipatory — the held breath before explosive motion. Professional automotive commercial quality, 8K, shallow depth of field with background bokeh, anamorphic lens flare, rich cinematic color grading with deep blacks and warm golden highlights",
            4.0  # guidance (unchanged)
        ]

    elif node["id"] == 6:  # FLUX negative prompt
        node["widgets_values"] = [
            "blurry, low quality, distorted, ugly, bad anatomy, extra limbs, missing fingers, deformed hands, watermark, text, oversaturated, cartoon, 3d render, painting, disfigured, out of focus, bad lighting, motion blur, moving car, driving, wheels spinning, speed lines, smoke from tires, motion, action shot, people, driver, person"
        ]

    # === 2. Update LTX prompts for realistic car motion ===
    elif node["id"] == 13:  # LTX positive prompt
        node["widgets_values"] = [
            "The hypercar remains perfectly still for a breath, parked diagonally on the wet coastal road. Then the LED headlights flicker on sharply, casting crisp white beams through the golden dusk air. The engine ignites with a deep visceral growl that builds into a thunderous roar, the quad exhaust tips beginning to glow with heat. The car body shudders subtly as power surges through the drivetrain. After a tense pause, the rear tires suddenly bite into the wet asphalt and the hypercar launches forward with brutal acceleration — a violent spray of water mist and gravel erupts from the rear wheel wells. The vehicle rockets away from the camera along the winding coastal road, its sculpted rear diffuser and active spoiler visible as it shrinks into the distance. The taillights streak into crimson light trails against the deepening sunset. Water droplets slowly settle back onto the pavement in the car's wake. The camera remains locked in its low position, steady and cinematic. Continuous coherent motion, no morphing or warping, physically realistic vehicle dynamics"
        ]

    elif node["id"] == 14:  # LTX negative prompt
        node["widgets_values"] = [
            "blurry, jittery, low quality, distorted, bad motion, static image, still frame, no movement, ugly, watermark, text, stop motion, choppy, jerky, teleporting, morphing, warping, flickering, inconsistent, ghosting, motion blur, frame skip, car disappearing, sudden cut, glitch, deformation, rubber tires, floating car, hovering, cartoon physics, unreal engine, CGI look, plastic material, toy car"
        ]

    # === 3. Optimize LTX parameters ===
    elif node["id"] == 17:  # NAG - slightly bolder for realistic motion
        node["widgets_values"] = [13.0, 0.35, 2.0]  # scale=13, alpha=0.35, tau=2.0

    elif node["id"] == 22:  # I2V - more freedom for explosive launch
        node["widgets_values"] = [0.72, False]  # strength=0.72 (was 0.80)

    elif node["id"] == 24:  # LTX Scheduler - more steps, optimized shifts
        node["widgets_values"] = [30, 2.05, 0.95, True, 0.08]  # 30 steps, terminal=0.08

    elif node["id"] == 28:  # Decoder Noise - slightly more texture
        node["widgets_values"] = [0.06, 0.03, 42, "randomize"]  # timestep=0.06, scale=0.03

    elif node["id"] == 20:  # Preprocess - slightly less compression for car details
        node["widgets_values"] = [30]  # img_compression=30 (was 35)

# === 4. Add LTXVApplySTG node (new id=33) ===
# This patches the LTX transformer blocks for STG guidance
stg_node = {
    "id": 33,
    "type": "LTXVApplySTG",
    "pos": [50, 910],
    "size": [340, 82],
    "flags": {},
    "order": 15,
    "mode": 0,
    "title": "33. STG 时空跳跃引导 (blocks 14,19)",
    "inputs": [
        {"name": "model", "type": "MODEL", "link": 36},  # from UNETLoader (16)
        {"name": "block_indices", "type": "STRING", "widget": {"name": "block_indices"}, "link": None}
    ],
    "outputs": [
        {"name": "model", "type": "MODEL", "links": [37]}  # to NAG (17)
    ],
    "properties": {"Node name for S&R": "LTXVApplySTG"},
    "widgets_values": ["14, 19"]
}
wf["nodes"].append(stg_node)

# === 5. Replace CFGGuider (23) with STGGuider ===
for node in wf["nodes"]:
    if node["id"] == 23:
        node["type"] = "STGGuider"
        node["title"] = "23. STG引导 (stg=1.5, rescale=0.7)"
        # Add STG-specific inputs while keeping model/pos/neg/cfg
        new_inputs = [
            {"name": "model", "type": "MODEL", "link": 21},
            {"name": "positive", "type": "CONDITIONING", "link": 17},
            {"name": "negative", "type": "CONDITIONING", "link": 18},
            {"name": "cfg", "type": "FLOAT", "widget": {"name": "cfg"}, "link": None},
            {"name": "stg", "type": "FLOAT", "widget": {"name": "stg"}, "link": None},
            {"name": "rescale", "type": "FLOAT", "widget": {"name": "rescale"}, "link": None},
        ]
        node["inputs"] = new_inputs
        node["widgets_values"] = [1.0, 1.5, 0.7]  # cfg=1.0, stg=1.5, rescale=0.7
        node["properties"]["Node name for S&R"] = "STGGuider"
        break

# === 6. Update links ===
# Old link 19: [19, 16, 0, 17, 0, "MODEL"] — UNET→NAG
# New: UNET→STGApply (link 36) + STGApply→NAG (link 37)
# Find and modify link 19
new_links = []
for link in wf["links"]:
    link_id = link[0]
    if link_id == 19:
        # Change to point from STGApply(33) to NAG(17)
        new_links.append([37, 33, 0, 17, 0, "MODEL"])
    else:
        new_links.append(link)

# Add new link: UNET(16) → STGApply(33)
new_links.append([36, 16, 0, 33, 0, "MODEL"])
wf["links"] = new_links

# === 7. Update metadata ===
wf["last_node_id"] = 33
wf["last_link_id"] = 37

# === 8. Add STG group ===
wf["groups"].append({
    "id": 9,
    "title": "🎯 STG 时空引导 — LTXVApplySTG + STGGuider (stg=1.5)",
    "bounding": [30, 890, 360, 200],
    "color": "#e74c3c"
})

# === 9. Update node titles for clarity ===
for node in wf["nodes"]:
    if node["id"] == 13:
        node["title"] = "13. LTX 正向Prompt (跑车启动→弹射，详细运动)"
    elif node["id"] == 14:
        node["title"] = "14. LTX 负向Prompt (排除CGI/虚幻/变形)"
    elif node["id"] == 17:
        node["title"] = "17. NAG引导 (scale=13, alpha=0.35, tau=2.0)"
    elif node["id"] == 22:
        node["title"] = "22. 图片→视频条件 (strength=0.72, 更多弹射自由)"
    elif node["id"] == 24:
        node["title"] = "24. LTX 调度器 (30步, terminal=0.08)"

# Write optimized workflow
output_path = "/home/steven/comfy/flux_ltx_i2v_optimized.json"
with open(output_path, "w") as f:
    json.dump(wf, f, indent=2)

print("✅ Optimized workflow written to:", output_path)
print(f"   Nodes: {len(wf['nodes'])} (added STG)")
print(f"   Links: {len(wf['links'])}")
print()
print("Key changes:")
print("  + LTXVApplySTG (node 33): blocks 14,19 for perturbed pass")
print("  ± STGGuider (node 23): stg=1.5, rescale=0.7, cfg=1.0")
print("  ± NAG: scale 12→13, alpha 0.30→0.35, tau 2.5→2.0")
print("  ± I2V strength: 0.80→0.72 (more launch freedom)")
print("  ± LTX steps: 25→30")
print("  ± DecoderNoise: timestep 0.05→0.06, scale 0.025→0.03")
print("  ± Terminal sigma: 0.1→0.08 (cleaner finish)")
print("  ± Preprocess compression: 35→30 (sharper car details)")
print("  ± Car-specific prompts (hypercar launch scene)")
