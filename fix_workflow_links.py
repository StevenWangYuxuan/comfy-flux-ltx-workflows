#!/usr/bin/env python3
"""Fix: properly update node input/output link references when adding STG node."""

import json

with open("/home/steven/comfy/flux_ltx_i2v_optimized.json") as f:
    wf = json.load(f)

# === Fix 1: Remap node 16 output link 19 -> 36 ===
# UNETLoader(16) now feeds STGApply(33), not NAG(17) directly
for n in wf["nodes"]:
    if n["id"] == 16:  # LTX UNETLoader
        # Change output link reference: old link 19 -> new link 36
        for out in n["outputs"]:
            if 19 in out.get("links", []):
                out["links"] = [36 if x == 19 else x for x in out["links"]]
                print(f"  Node 16 output: links {out['links']}")

    # === Fix 2: Remap node 17 input link 19 -> 37 ===
    # NAG(17) now receives from STGApply(33), not UNETLoader(16)
    elif n["id"] == 17:  # LTX2_NAG
        for inp in n["inputs"]:
            if inp["link"] == 19:
                inp["link"] = 37
                print(f"  Node 17 input '{inp['name']}': link 19 -> 37")

# === Fix 3: Ensure STGApply node inputs/outputs are correct ===
for n in wf["nodes"]:
    if n["id"] == 33:  # LTXVApplySTG
        # Input should reference link 36 (from UNET 16)
        n["inputs"][0]["link"] = 36
        # Output should reference link 37 (to NAG 17)
        n["outputs"][0]["links"] = [37]
        print(f"  Node 33 input link: {n['inputs'][0]['link']}")
        print(f"  Node 33 output links: {n['outputs'][0]['links']}")

# === Fix 4: Rebuild links array correctly ===
# Keep FLUX links (1-14) as-is
# LTX links:
#   13: CLIP(12) -> PosEncode(13)
#   14: CLIP(12) -> NegEncode(14)
#   15: PosEncode(13) -> Conditioning(15)
#   16: NegEncode(14) -> Conditioning(15)
#   17: Conditioning(15) pos -> STGGuider(23)
#   18: Conditioning(15) neg -> STGGuider(23)
#   36: UNET(16) -> STGApply(33)    [NEW]
#   37: STGApply(33) -> NAG(17)     [NEW]
#   20: NAG(17) -> LoRA(18)
#   21: LoRA(18) -> STGGuider(23)
#   22: VAE(19) -> I2V(22)
#   23: VAE(19) -> DecoderNoise(28)
#   24: Preprocess(20) -> I2V(22)
#   25: Latent(21) -> I2V(22)
#   26: I2V(22) -> Sampler(27)
#   27: STGGuider(23) -> Sampler(27)
#   28: Scheduler(24) -> Sampler(27)
#   29: SamplerSelect(25) -> Sampler(27)
#   30: Noise(26) -> Sampler(27)
#   31: Sampler(27) -> TiledDecode(29)
#   32: DecoderNoise(28) -> TiledDecode(29)
#   33: TiledDecode(29) -> CreateVideo(30)
#   34: CreateVideo(30) -> SaveVideo(32)
#   35: TiledDecode(29) -> Preview(31)

new_links = [
    # FLUX half (unchanged)
    [1, 3, 0, 5, 0, "CLIP"],
    [2, 3, 0, 6, 0, "CLIP"],
    [3, 4, 0, 10, 1, "VAE"],
    [4, 5, 0, 9, 1, "CONDITIONING"],
    [5, 6, 0, 7, 0, "CONDITIONING"],
    [6, 7, 0, 9, 2, "CONDITIONING"],
    [7, 8, 0, 9, 3, "LATENT"],
    [8, 1, 0, 2, 0, "MODEL"],
    [9, 2, 0, 9, 0, "MODEL"],
    [10, 9, 0, 10, 0, "LATENT"],
    [11, 10, 0, 11, 0, "IMAGE"],
    [12, 10, 0, 20, 0, "IMAGE"],
    # LTX half
    [13, 12, 0, 13, 0, "CLIP"],
    [14, 12, 0, 14, 0, "CLIP"],
    [15, 13, 0, 15, 0, "CONDITIONING"],
    [16, 14, 0, 15, 1, "CONDITIONING"],
    [17, 15, 0, 23, 1, "CONDITIONING"],  # pos -> STGGuider
    [18, 15, 1, 23, 2, "CONDITIONING"],  # neg -> STGGuider
    [36, 16, 0, 33, 0, "MODEL"],         # UNET -> STGApply [NEW]
    [37, 33, 0, 17, 0, "MODEL"],         # STGApply -> NAG [NEW]
    [20, 17, 0, 18, 0, "MODEL"],         # NAG -> LoRA
    [21, 18, 0, 23, 0, "MODEL"],         # LoRA -> STGGuider
    [22, 19, 0, 22, 0, "VAE"],
    [23, 19, 0, 28, 0, "VAE"],
    [24, 20, 0, 22, 1, "IMAGE"],
    [25, 21, 0, 22, 2, "LATENT"],
    [26, 22, 0, 27, 4, "LATENT"],        # I2V -> Sampler (slot 4)
    [27, 23, 0, 27, 1, "GUIDER"],        # STGGuider -> Sampler
    [28, 24, 0, 27, 3, "SIGMAS"],
    [29, 25, 0, 27, 2, "SAMPLER"],
    [30, 26, 0, 27, 0, "NOISE"],
    [31, 27, 0, 29, 1, "LATENT"],
    [32, 28, 0, 29, 0, "VAE"],
    [33, 29, 0, 30, 0, "IMAGE"],
    [34, 30, 0, 32, 0, "VIDEO"],
    [35, 29, 0, 31, 0, "IMAGE"],
]

wf["links"] = new_links

# === Fix 5: Also update all node input/output link references comprehensively ===
for n in wf["nodes"]:
    nid = n["id"]

    # Update all input links to match the new links array
    for inp in n["inputs"]:
        if inp.get("link") is not None:
            old_link = inp["link"]
            # Find what new link goes to this node/input
            found = False
            for link in new_links:
                if link[3] == nid and link[4] == n["inputs"].index(inp):
                    inp["link"] = link[0]
                    if link[0] != old_link:
                        print(f"  Node {nid} input '{inp['name']}': link {old_link} -> {link[0]}")
                    found = True
                    break
            if not found:
                # Input without link is fine (widget value)
                pass

    # Update all output link references
    for out in n.get("outputs", []):
        out_idx = n["outputs"].index(out)
        expected_links = []
        for link in new_links:
            if link[1] == nid and link[2] == out_idx:
                expected_links.append(link[0])
        if expected_links:
            out["links"] = expected_links

# Write
with open("/home/steven/comfy/flux_ltx_i2v_optimized.json", "w") as f:
    json.dump(wf, f, indent=2)

# Final validation
print("\n=== VALIDATION ===")
errors = []
for n in wf["nodes"]:
    nid = n["id"]
    for inp in n["inputs"]:
        link_id = inp.get("link")
        if link_id is not None and link_id != 0:
            # Check this link exists in links array
            found = False
            for link in wf["links"]:
                if link[0] == link_id and link[3] == nid:
                    found = True
                    break
            if not found:
                errors.append(f"  ❌ Node {nid} input '{inp['name']}' references link {link_id} which doesn't exist or doesn't connect to this node")

for link in wf["links"]:
    lid, src, src_slot, dst, dst_slot, typ = link
    # Check source node has this output
    src_node = next((n for n in wf["nodes"] if n["id"] == src), None)
    if src_node is None:
        errors.append(f"  ❌ Link {lid}: source node {src} not found")
    # Check dest node has this input
    dst_node = next((n for n in wf["nodes"] if n["id"] == dst), None)
    if dst_node is None:
        errors.append(f"  ❌ Link {lid}: dest node {dst} not found")
    # Check link ID is in source node's output links
    if src_node:
        expected_in_output = False
        for out in src_node.get("outputs", []):
            if lid in out.get("links", []):
                expected_in_output = True
        if not expected_in_output:
            errors.append(f"  ❌ Link {lid}: not referenced in source node {src} outputs")
    # Check link ID is in dest node's input
    if dst_node:
        dst_input = dst_node["inputs"][dst_slot]
        if dst_input.get("link") != lid:
            errors.append(f"  ❌ Link {lid}: dest node {dst} slot {dst_slot} has link={dst_input.get('link')}, expected {lid}")

if errors:
    print("ERRORS FOUND:")
    for e in errors:
        print(e)
else:
    print("✅ All links and node references are consistent!")
    print(f"   {len(wf['nodes'])} nodes, {len(wf['links'])} links")
