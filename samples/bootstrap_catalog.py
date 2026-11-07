"""
Bootstrap Script for RankForge:
1. Generates 1,000 synthetic catalog items across 8 diverse categories.
2. Batches item embeddings through ItemTower and ingests them into the in-memory vector index.
3. Simulates realistic user retrieval queries across diverse customer personas.
4. Benchmarks retrieval latency (p50, p95, p99) over the 1,000+ item catalog.
"""

import os
import random
import sys
import time
from typing import List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.item import Item
from app.models.recommendation import RecommendationRequest
from app.models.user import UserProfile
from app.services.indexer import default_indexer
from app.services.item_tower import item_tower
from app.services.user_tower import user_tower

# Seed for reproducible bootstrap data
random.seed(42)

CATEGORIES_DATA = {
    "Gaming": {
        "titles": [
            "Pro Wireless Mechanical Gaming Keyboard RGB",
            "Ultra-Lightweight 26K DPI Optical Gaming Mouse",
            "Immersive 7.1 Surround Sound Gaming Headset",
            "Curved 34-Inch 165Hz Ultrawide Gaming Monitor",
            "Ergonomic Memory Foam Esports Gaming Chair",
            "Next-Gen High-Speed Gaming Console 1TB SSD",
            "Low-Friction Speed Extended Cloth Mousepad",
            "Dual-Sense Wireless Gamepad Controller",
            "RGB Desktop Microphone with Tap-to-Mute",
            "External Game Capture Card 4K 60FPS HDR",
        ],
        "tags": ["gaming", "rgb", "esports", "peripherals", "wireless", "pc", "console", "ultra-fast"],
        "price_range": (39.99, 899.99),
    },
    "Electronics": {
        "titles": [
            "Flagship 6.7-inch OLED 5G Smartphone 256GB",
            "Ultra-Slim 14-inch Laptop Intel Core i7 32GB RAM",
            "Active Noise-Cancelling Over-Ear Studio Headphones",
            "Waterproof Portable Bluetooth 5.3 Speaker",
            "4K UHD Smart OLED TV with Dolby Vision",
            "Fast Magnetic Wireless Charging Pad 15W",
            "Mirrorless Full-Frame Digital Camera 24MP",
            "USB-C Multi-Port Docking Station 10-in-1",
            "High-Capacity 25,000mAh Power Bank 65W PD",
            "True Wireless In-Ear Earbuds with ANC",
        ],
        "tags": ["electronics", "audio", "smart", "bluetooth", "wireless", "hardware", "portable", "premium"],
        "price_range": (29.99, 1499.99),
    },
    "Home & Kitchen": {
        "titles": [
            "Stainless Steel Convection Air Fryer Oven",
            "Programmable Bean-to-Cup Espresso Machine",
            "Precision Induction Cooktop with Temperature Probe",
            "Japanese Damascus Steel Chef's Knife 8-inch",
            "Automatic Self-Emptying Robot Vacuum & Mop",
            "Double-Wall Vacuum Insulated French Press",
            "Cast Iron Enamelled Dutch Oven 6-Quart",
            "Smart Wi-Fi Food Thermometer with Dual Probes",
            "Compact Cold Press Slow Juicer Extractor",
            "Silicone Heat-Resistant Kitchen Utensil Set",
        ],
        "tags": ["kitchen", "cooking", "appliances", "coffee", "chef", "home", "dining", "culinary"],
        "price_range": (19.99, 699.99),
    },
    "Sports & Fitness": {
        "titles": [
            "Adjustable Quick-Select Dumbbell Set 5-52 lbs",
            "Breathable Lightweight Trail Running Shoes",
            "Non-Slip High-Density Natural Yoga Mat 6mm",
            "Smart GPS Multisport Fitness Watch Heart Rate",
            "Foldable Magnetic Indoor Rowing Machine",
            "Deep Tissue Percussion Muscle Massage Gun",
            "Insulated Stainless Steel Water Bottle 32oz",
            "Heavy-Duty Elastic Workout Resistance Bands",
            "Speed Jump Rope with Ball Bearings & Steel Cable",
            "Gym Duffel Bag with Shoe Compartment",
        ],
        "tags": ["fitness", "sports", "running", "workout", "gym", "outdoor", "recovery", "cardio"],
        "price_range": (14.99, 599.99),
    },
    "Books & Literature": {
        "titles": [
            "The Machine Learning Engineering Handbook 2nd Edition",
            "Chronicles of the Deep Cosmos: Hard Sci-Fi Epic",
            "Designing High-Performance Distributed Systems",
            "The Psychology of Decision Making & Behavioral Economics",
            "Algorithmic Thinking: Modern Data Structures in Action",
            "Quiet Mind, High Performance: The Stoic's Guide",
            "Deep Learning Architectures: From Perceptrons to Transformers",
            "The Great Galactic Journey: Space Opera Saga",
            "Clean Architecture & Microservices Best Practices",
            "Mastering Large-Scale Candidate Retrieval Systems",
        ],
        "tags": ["books", "programming", "science", "engineering", "sci-fi", "literature", "education", "economics"],
        "price_range": (12.99, 79.99),
    },
    "Fashion & Apparel": {
        "titles": [
            "Merino Wool Breathable Thermal Crewneck Sweater",
            "Water-Resistant All-Weather Commuter Parka Jacket",
            "Classic Fit Denim Jeans Stretch Selvedge",
            "Minimalist Chronograph Automatic Wristwatch",
            "Full-Grain Italian Leather Everyday Belt",
            "Vintage Polarized UV400 Sunglasses Acetate",
            "Premium Combed Cotton Crew Socks 6-Pack",
            "Casual Canvas Minimalist Low-Top Sneakers",
            "Waterproof Laptop Commuter Backpack 25L",
            "Tailored Wool Blend Slim-Fit Dress Blazer",
        ],
        "tags": ["fashion", "clothing", "apparel", "style", "menswear", "footwear", "accessories", "minimalist"],
        "price_range": (19.99, 349.99),
    },
    "Smart Home & IoT": {
        "titles": [
            "Smart Video Doorbell Camera 2K HDR Night Vision",
            "Smart Home Color-Changing LED Light Strip 16ft",
            "Learning Smart Thermostat with Room Sensors",
            "Motorized Smart Roller Shades Zigbee Compatible",
            "Keyless Fingerprint Touchscreen Deadbolt Smart Lock",
            "Smart Indoor Air Quality Monitor & CO2 Sensor",
            "Wi-Fi Mesh 6E Router Tri-Band High Throughput",
            "Smart Plug Mini with Real-Time Energy Monitoring",
            "Weather-Resistant Outdoor Security Spotlight Camera",
            "Voice-Controlled Smart Display Hub 10-inch",
        ],
        "tags": ["smarthome", "iot", "security", "lighting", "wifi", "automation", "connected", "energy"],
        "price_range": (24.99, 399.99),
    },
    "Beauty & Personal Care": {
        "titles": [
            "Hydrating Hyaluronic Acid & Vitamin C Face Serum",
            "Sonic Electric Toothbrush 40,000 VPM with Travel Case",
            "Ionic Hair Dryer Fast-Drying Brushless Motor",
            "Nourishing Organic Argan Oil Hair Treatment",
            "Precision Rotary Electric Shaver Waterproof Wet/Dry",
            "Mineral Sunscreen SPF 50 Broad Spectrum Matte",
            "Calming Lavender & Chamomile Aromatherapy Diffuser",
            "Exfoliating Dead Sea Salt Body Scrub 16oz",
            "Ceramide Barrier Repair Moisturizing Face Cream",
            "Beard Grooming Kit with Sandalwood Oil & Comb",
        ],
        "tags": ["beauty", "skincare", "grooming", "wellness", "personalcare", "organic", "spa", "health"],
        "price_range": (9.99, 189.99),
    },
}


def generate_synthetic_catalog(total_items: int = 1000) -> List[Item]:
    """Generates synthetic catalog items with realistic attributes."""
    items: List[Item] = []
    categories = list(CATEGORIES_DATA.keys())
    items_per_category = total_items // len(categories)

    item_counter = 1
    for cat in categories:
        data = CATEGORIES_DATA[cat]
        titles = data["titles"]
        tags_pool = data["tags"]
        p_min, p_max = data["price_range"]

        for i in range(items_per_category):
            base_title = titles[i % len(titles)]
            # Add synthetic model / variant designation for variety
            variant = f"Series {chr(65 + (i % 26))}{i // 26 + 1}"
            title = f"{base_title} - {variant}" if i >= len(titles) else base_title

            item_id = f"itm_{cat[:3].lower()}_{item_counter:04d}"
            item_counter += 1

            # Sample 2-4 tags from category pool
            num_tags = random.randint(2, 4)
            item_tags = list(set(random.sample(tags_pool, num_tags)))

            price = round(random.uniform(p_min, p_max), 2)
            popularity = round(random.uniform(0.1, 0.99), 3)
            rating = round(random.uniform(3.5, 5.0), 2)
            in_stock = random.random() > 0.05  # 95% in stock

            items.append(
                Item(
                    item_id=item_id,
                    title=title,
                    category=cat,
                    tags=item_tags,
                    price=price,
                    popularity_score=popularity,
                    rating=rating,
                    in_stock=in_stock,
                    description=f"{title}. Ideal for enthusiasts seeking high quality in {cat}.",
                    metadata={"catalog_batch": "bootstrap_v1", "warranty_months": random.choice([12, 24, 36])},
                )
            )

    # Fill any remaining items to reach exact total_items
    while len(items) < total_items:
        cat = random.choice(categories)
        data = CATEGORIES_DATA[cat]
        title = f"{random.choice(data['titles'])} (Limited Edition #{item_counter})"
        item_id = f"itm_{cat[:3].lower()}_{item_counter:04d}"
        item_counter += 1
        items.append(
            Item(
                item_id=item_id,
                title=title,
                category=cat,
                tags=random.sample(data["tags"], 3),
                price=round(random.uniform(*data["price_range"]), 2),
                popularity_score=0.8,
                rating=4.5,
                in_stock=True,
                description=f"Special edition catalog item for {cat}.",
            )
        )

    return items


def bootstrap_index(total_items: int = 1000) -> Tuple[int, float]:
    """Generates synthetic items, embeds them via ItemTower, and adds to default_indexer."""
    print(f"[*] Generating {total_items} synthetic catalog items across 8 categories...")
    items = generate_synthetic_catalog(total_items)

    print(f"[*] Projecting items through ItemTower (64-dim L2-normalized embeddings)...")
    start = time.perf_counter()
    embeddings = item_tower.predict_batch(items)
    tower_latency = (time.perf_counter() - start) * 1000.0
    print(f"    -> ItemTower batch inference latency: {tower_latency:.2f} ms ({tower_latency / len(items):.3f} ms/item)")

    print(f"[*] Ingesting into VectorIndexer [backend: {default_indexer.backend_name}]...")
    start_ingest = time.perf_counter()
    ids = [item.item_id for item in items]
    metadatas = [item.model_dump() for item in items]
    inserted = default_indexer.add_items(embeddings, ids, metadatas=metadatas)
    ingest_latency = (time.perf_counter() - start_ingest) * 1000.0

    print(f"    -> Ingestion latency: {ingest_latency:.2f} ms")
    print(f"[+] Total items in index: {default_indexer.total_items}\n")
    return inserted, tower_latency + ingest_latency


def simulate_persona_recommendations():
    """Simulates realistic recommendation requests for distinct user personas."""
    personas = [
        {
            "name": "Alex - Esports & PC Gaming Enthusiast",
            "profile": UserProfile(
                user_id="usr_alex_01",
                device="desktop",
                region="US",
                age_bucket="18-24",
                historical_interactions=["itm_gam_0001", "itm_gam_0003"],
                category_affinities={"Gaming": 0.95, "Electronics": 0.40},
                search_query="wireless mechanical keyboard optical switch",
                activity_level=0.9,
            ),
            "top_k": 5,
        },
        {
            "name": "Sophia - Home Barista & Culinary Chef",
            "profile": UserProfile(
                user_id="usr_sophia_02",
                device="mobile",
                region="EU",
                age_bucket="25-34",
                historical_interactions=[],
                category_affinities={"Home & Kitchen": 0.92},
                search_query="bean to cup espresso machine grinder",
                activity_level=0.7,
            ),
            "top_k": 5,
        },
        {
            "name": "Marcus - Marathon Runner & Fitness Athlete",
            "profile": UserProfile(
                user_id="usr_marcus_03",
                device="mobile",
                region="US",
                age_bucket="25-34",
                historical_interactions=["itm_spo_0380"],
                category_affinities={"Sports & Fitness": 0.90},
                search_query="gps trail running watch heart rate",
                activity_level=0.85,
            ),
            "top_k": 5,
        },
        {
            "name": "Elena - Computer Scientist & Sci-Fi Reader",
            "profile": UserProfile(
                user_id="usr_elena_04",
                device="desktop",
                region="EU",
                age_bucket="35-50",
                historical_interactions=["itm_boo_0510"],
                category_affinities={"Books & Literature": 0.95},
                search_query="distributed systems machine learning algorithms",
                activity_level=0.6,
            ),
            "top_k": 5,
        },
        {
            "name": "Taylor - Cold-Start / Exploratory User",
            "profile": UserProfile(
                user_id="usr_taylor_05",
                device="mobile",
                region="APAC",
                age_bucket="25-34",
                historical_interactions=[],
                category_affinities={"Smart Home & IoT": 0.70},
                search_query=None,
                activity_level=0.3,
            ),
            "top_k": 5,
        },
    ]

    print("=" * 80)
    print("SIMULATING REALISTIC TWO-TOWER CANDIDATE RETRIEVAL QUERIES")
    print("=" * 80)

    for p in personas:
        name = p["name"]
        user_prof: UserProfile = p["profile"]
        top_k = p["top_k"]

        print(f"\n>>> Persona: {name}")
        print(f"    User ID: {user_prof.user_id} | Device: {user_prof.device} | Region: {user_prof.region}")
        print(f"    Category Affinities: {user_prof.category_affinities}")
        print(f"    Search Query: '{user_prof.search_query or '(none)'}'")

        # 1. User Tower projection
        t0 = time.perf_counter()
        user_vec = user_tower.predict_one(user_prof)
        user_tower_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Vector Index Retrieval
        t1 = time.perf_counter()
        raw_candidates = default_indexer.query(
            vector=user_vec,
            top_k=top_k,
            filter_fn=lambda m: m.get("in_stock", True) and m.get("item_id") not in user_prof.historical_interactions,
        )
        retrieval_ms = (time.perf_counter() - t1) * 1000.0
        total_ms = user_tower_ms + retrieval_ms

        print(f"    Latency: UserTower={user_tower_ms:.3f}ms | Vector Retrieval={retrieval_ms:.3f}ms | Total={total_ms:.3f}ms")
        print(f"    Ranked Top-{len(raw_candidates)} Candidates:")
        print(f"    {'Rank':<5} {'Score':<8} {'Category':<18} {'Price':<9} {'Title'}")
        print("    " + "-" * 74)

        for rank, (item_id, score, meta) in enumerate(raw_candidates, start=1):
            title = meta.get("title", "")
            cat = meta.get("category", "")
            price = f"${meta.get('price', 0.0):.2f}"
            print(f"    #{rank:<4} {score:+.4f}  {cat:<18} {price:<9} {title[:38]}")


def benchmark_retrieval(num_queries: int = 500):
    """Benchmarks candidate retrieval latency over 500 queries."""
    print("\n" + "=" * 80)
    print(f"BENCHMARKING RETRIEVAL LATENCY OVER {num_queries} QUERIES ({default_indexer.total_items} items in index)")
    print("=" * 80)

    categories = list(CATEGORIES_DATA.keys())
    latencies = []

    for i in range(num_queries):
        cat = random.choice(categories)
        profile = UserProfile(
            user_id=f"bench_usr_{i}",
            category_affinities={cat: random.uniform(0.6, 1.0)},
            activity_level=random.uniform(0.2, 0.9),
        )
        vec = user_tower.predict_one(profile)

        t0 = time.perf_counter()
        _ = default_indexer.query(vec, top_k=10)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)

    import numpy as np

    arr = np.array(latencies)
    p50 = np.percentile(arr, 50)
    p90 = np.percentile(arr, 90)
    p95 = np.percentile(arr, 95)
    p99 = np.percentile(arr, 99)
    mean = np.mean(arr)

    print(f"Retrieval Latency Summary (Top-10 retrieval over {default_indexer.total_items} items):")
    print(f"  * p50 (Median) : {p50:.3f} ms")
    print(f"  * p90          : {p90:.3f} ms")
    print(f"  * p95          : {p95:.3f} ms")
    print(f"  * p99          : {p99:.3f} ms")
    print(f"  * Mean         : {mean:.3f} ms")
    print(f"  * Max          : {np.max(arr):.3f} ms")
    print("=" * 80)


if __name__ == "__main__":
    bootstrap_index(1000)
    simulate_persona_recommendations()
    benchmark_retrieval(500)
