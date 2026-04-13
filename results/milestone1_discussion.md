# Qualitative evaluation for BM25 and semantic search

In `notebooks/milestone1_exploration.ipynb`, we performed a qualitative evaluation of the BM25 and semantic search methods. This involved selecting a few queries and examining the retrieved results for each method. This analysis helped us understand the strengths and weaknesses of each approach and provided insights into how they might be improved in future iterations.

## Queries

We select a set of queries that are representative of the "Appliance" category in the dataset. These queries are designed to test the retrieval capabilities of both methods and to highlight any differences in their performance. We have chosen the queries based on their "difficulty" level and expected winning system.

| Query | Difficulty | Expected winner |
| --- | --- | --- |
| stainless steel dishwasher | Easy | BM25 |
| refrigerator water filter | Easy | BM25 |
| gas range 30 inch | Easy | BM25 |
| something to keep drinks cold in a dorm room | Medium | Semantic |
| appliance for washing dishes quietly at night | Medium | Semantic |
| replacement part to stop a dishwasher from leaking at the bottom | Medium | Semantic |
| small machine for drying clothes in an apartment | Medium | Semantic |
| best dishwasher for a small apartment under $800 | Complex | Neither |
| energy-efficient refrigerator for a family of four | Complex | Neither |
| reliable stove for frequent home cooking that is easy to clean | Complex | Neither |

## Results

We compare results of 5 out of 10 queries across both methods, showing the top five retrieved items for each query.

### Query 1: "stainless steel dishwasher"

#### BM25 search

1. [metadata] KitchenAid Superba Series KUDS30SXSS Fully Integrated Dishwasher, Stainless Steel  
2. [metadata] MLGB Stainless Steel Brushed Pattern Dishwasher Magnet Cover Panel Decal Home Appliance Art, Stainless Steel Fridge Door Cover Decals Magnetic, Black, Mobile Magnetic 23" x 26"  
3. [metadata] midea HS-209BESS Beer/Beverage Refrigerator and Dispenser, 5.7 Cubic Feet, Stainless Steel  
4. [review] stainless steel pitcher  
5. [metadata] GE CGS985SETSS Cafe 30" Stainless Steel Gas Sealed Burner Range - Convection

#### Semantic search

1. [metadata] KitchenAid Superba Series KUDS30SXSS Fully Integrated Dishwasher, Stainless Steel  
2. [metadata] Frigidaire Professional FPID2495QF Fully Integrated Dishwasher  
3. [metadata] ERP 809006501 Dishwasher Lower Seal  
4. [review] Finally, the branded product to clean my Miele!  
5. [metadata] Dishwasher Silverware Basket, Genuine Original Equipment Manufacturer Dishwasher Cutlery Basket Compatible with Kenmore, Whirlpool, Bosch, Maytag, KitchenAid, Samsung, GE, and more

#### Comments

BM25 performs better for this query since it captures the exact phrase “stainless steel” better, but several results drift to unrelated stainless steel items. Semantic search stays closer to the dishwasher category overall, though it is less precise about the stainless steel requirement.

### Query 2: "refrigerator water filter"

#### BM25 search

1. [metadata] Replacement for LG LFX28978ST/00 Refrigerator Water Filter - Compatible with LG LFX28978ST/00 Fridge Water Filter Cartridge  
2. [metadata] Crystala Filters LT1000PC Refrigerator Water Filter, Water Filter ADQ747935 Compatible with LT1000PC, LT1000PC/PCS, LT-1000PC, MDJ64844601, ADQ747935, ADQ74793504 Water Filter (4 Pack)  
3. [metadata] 4-Pack Replacement for LG LFX25974SW Refrigerator Water Filter - Compatible with LG 5231JA2002A, LT500P Fridge Water Filter Cartridge  
4. [metadata] Compatible with LT700P Refrigerator Water Filter, LG Water Filter Compatible with LT700P, ADQ36006101, ADQ36006102 and KENMORE 9690, 469690-3 Pack  
5. [metadata] 3-Pack Replacement for KitchenAid KSSC36FMS Refrigerator Water Filter - Compatible with KitchenAid 4396508, 4396509, 4396510 Fridge Water Filter Cartridge

#### Semantic search

1. [review] Wirlpool Water filter for Refrigerater  
2. [metadata] EveryDrop by Whirlpool Refrigerator Water Filter 3 (Pack of 3)  
3. [review] Original Frigidaire water filter that is easy to install  
4. [metadata] NEW GE Smart Water Filter BY-PASS Fitting Plug WR02X11705 Refrigerator MWF GFW  
5. [metadata] Aqua Fresh RPWF Refrigerator Water Filter Compatible with GE RPWF, RWF1063, RWF3600A, WSG-4, DWF-36, R-3600, MPF15350, OPFG3-RF300 (3 Pack)

#### Comments

BM25 performs better for this query because it directly matches the exact product phrase “refrigerator water filter” and returns highly relevant product listings. Semantic search also retrieves relevant items, but it mixes product listings with review documents and includes a less relevant item. Semantic search mainly underperforms BM25 by being broader than necessary.

### Query 3: "something to keep drinks cold in a dorm room"

#### BM25 search

1. [review] Love to use a lot everyday!!  
2. [review] Works Well  
3. [review] Great product  
4. [metadata] Compact Laundry Dryer, 1400W Electric Portable Clothes Dryer, 9LBS Laundry Dryer, Premium Compact Tumbler Stainless Steel Laundry Dryer with Exhaust Pipe, LCD Panel Dryer for Apartments, Home, Dorm (Whit-9LBs)  
5. [metadata] Repairwares Washing Machine Cold Water Inlet Valve Assembly 422244 00422244 1105556 WV2244 PS3462925 PS8713229 for Select Bosch Washer Models

#### Semantic search

1. [review] Love to use a lot everyday!!  
2. [review] Pellet Ice!!  
3. [metadata] NewAir NBC126SS02 Beverage Refrigerator and Cooler, Holds up to 126 Cans, Cools Down to 37 Degrees Perfect for Beer Wine or Soda, 126 Can, Silver, 126 Can (Renewed)  
4. [metadata] G.a HOMEFAVOR Cold Brew Coffee Infuser 64oz (2 Quart), Stainless Steel Filter Kit for Wide Mason Jar and Iced Tea Maker at Home  
5. [review] Works Well

#### Comments

Semantic search performs better for this query because it captures the underlying meaning of “keep drinks cold” retrieves a relevant beverage refrigerator. BM25 struggles because the query is phrased as natural languages need rather than a specific product name, so it overmatches words like “cold” and “dorm”. This is a clear case where BM25 fails but semantic search partially succeeds, but it also fails in some results by returning loosely related items such as an ice machine review and a cold brew coffee infuser rather than consistently retrieving mini fridges or beverage refrigerators.

### Query 4: "replacement part to stop a dishwasher from leaking at the bottom"

#### BM25 search

1. [review] It leaking on bottom but those company is out of business so can’t replace!!  
2. [metadata] HASMX W11157084 WP8561996 Dishwasher Upper Rack Wheel Mount Replacement Part for Whirlpool Maytag - Replaces Part Numbers W11157084, AP6285708, WP8561996, 8561996, PS973972, B0050O2HIO, B00JLMM1V4 (2)  
3. [metadata] 154567702 Dishwasher Lower Wash Arm Assembly for Kenmore Electrolux Dishwasher Bottom Lower Spray Arm 5304518927, AP6810011,154567701…  
4. [review] Works well  
5. [review] This holds 14 eggs nicely in the fridge.

#### Semantic search

1. [review] It leaking on bottom but those company is out of business so can’t replace!!  
2. [metadata] ERP 809006501 Dishwasher Lower Seal  
3. [review] Works fits our dishwasher  
4. [metadata] Frigidaire Professional FPID2495QF Fully Integrated Dishwasher  
5. [metadata] 154567702 Dishwasher Lower Wash Arm Assembly for Kenmore Electrolux Dishwasher Bottom Lower Spray Arm 5304518927, AP6810011,154567701…

### Comments

Semantic search performs better for this query because it captures user's intention more effectively and retrieves a highly relevant part, “Dishwasher Lower Seal,” which directly matches the idea of stopping a bottom leak. BM25 fails noticeably by returning an unrelated items, none of which are useful for fixing a leaking dishwasher. However, semantic search still ranks a generic review first and includes a full dishwasher product instead of only replacement parts.

### Query 5: "energy-efficient refrigerator for a family of four"

#### BM25 search

1. [review] Four Stars  
2. [review] Love to use a lot everyday!!  
3. [review] I love my little ice maker  
4. [metadata] ASSINAI Portable Mini Washing Machine, Ultrasonic Washing Machine 3 In 1 Dishwasher Mini Light, Convenient for Travel, Family Business Travel, USB  
5. [metadata] WATERMOON Egg Dispenser For Refrigerator - 24 Count Rolling Egg Holder For Refrigerator Egg Storage Container Egg Drawer For Refrigerator Egg Tray For Refrigerator With 4 Spoons,Grey

#### Semantic search

1. [review] Works Well  
2. [metadata] Maytag MFX2570AEB Ice2O 25 Cu. Ft. Black French Door Refrigerator - Energy Star  
3. [metadata] GE GSS25GGHWW Side Refrigerator  
4. [metadata] midea HS-209BESS Beer/Beverage Refrigerator and Dispenser, 5.7 Cubic Feet, Stainless Steel  
5. [review] Waste of money!

#### Comments

Semantic search performs better for this query because it retrieves actual refrigerator products, while BM25 fails by only returning items with overlapping words. Semantic search still does not explicitly satisfy the “for a family of four” requirement, and some results are only loosely related, such as a beverage refrigerator and an irrelevant review. This suggests that the complex queries would likely benefit from reranking or LLM-based reasoning.

## Summary and insights

BM25 performs best on clear keyword-based queries where the exact product type or attribute appears directly in the text, but it often fails when queries are phrased as natural-language needs or when individual keywords cause irrelevant matches.

Semantic search performs better on intent-based queries by capturing meaning beyond exact word overlap, but it can also be too broad, sometimes returning related but incorrect reviews or products that miss an important constraint.

Both methods struggle on more complex queries that involve preferences, tradeoffs, or inferred requirements, such as budget limits, household size, product quality, or ease of cleaning. More advanced methods such as reranking or RAG could further improve the retrieval results. Reranking could improve precision by prioritizing results that satisfy the full query intent, while LLM-based methods could better interpret complex needs like “for a family of four”.
