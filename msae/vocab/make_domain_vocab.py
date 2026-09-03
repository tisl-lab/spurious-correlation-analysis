"""Build a visually-grounded domain vocabulary for SAE concept naming.

Produces a curated list of concrete terms relevant to Waterbirds-style photos:
birds, animals, landscapes/nature, physical objects, colors/materials/textures,
and the 200 CUB-200-2011 bird species that Waterbirds is built from.

The vocabulary is consumed by re-embedding it through CLIP's text encoder
(msae/precompute_activations.py), NOT by reusing the fixed DISECT embeddings —
so words do NOT need to appear in clip_disect_20k.txt. We therefore keep every
curated term, and only *report* DISECT coverage for information. The CUB species
in particular are almost entirely absent from DISECT, which is exactly why they
are worth adding.

CUB species are read live from the Waterbirds metadata.csv so the list is the
exact 200 classes this dataset uses (no hand-transcription).

Usage:
    python msae/vocab/make_domain_vocab.py
"""

import csv
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SOURCE = os.path.join(SCRIPT_DIR, "clip_disect_20k.txt")
OUTPUT = os.path.join(SCRIPT_DIR, "waterbirds_domain_vocab.txt")
METADATA = os.path.join(REPO_ROOT, "data", "waterbirds", "metadata.csv")

# ── Candidate terms per category ────────────────────────────────────────────

BIRDS = """
bird birds duck ducks eagle eagles hawk hawks owl owls penguin penguins swan
sparrow sparrows robin cardinal cardinals crow crows pigeon pigeons gull gulls
heron crane cranes flamingo peacock parrot parrots canary finch finches raven
ravens falcon falcons dove doves turkey turkeys goose chicken chickens rooster
hen hens quail pelican stork woodpecker kingfisher nightingale ostrich
hummingbird wren magpie starling jay blackbird oriole warbler pheasant
partridge cormorant albatross puffin toucan macaw cockatoo parakeet feather
feathers beak bill wing wings nest nests plumage flock talon egg eggs aviary
poultry waterfowl songbird seabird
""".split()

ANIMALS = """
animal animals cat cats dog dogs horse horses cow cows cattle sheep goat goats
pig pigs deer bear bears wolf wolves fox foxes rabbit rabbits squirrel squirrels
mouse mice rat rats lion lions tiger tigers elephant elephants monkey monkeys
giraffe zebra zebras kangaroo koala panda leopard cheetah rhino rhinoceros hippo
hippopotamus crocodile crocodiles alligator snake snakes lizard lizards turtle
turtles frog frogs toad fish fishes shark sharks whale whales dolphin dolphins
seal seals otter otters beaver beavers raccoon moose elk buffalo bison camel
camels donkey donkeys mule pony ponies lamb lambs calf calves foal kitten kittens
puppy puppies cub cubs insect insects butterfly butterflies bee bees ant ants
spider spiders beetle beetles dragonfly moth mosquito worm worms snail snails
crab crabs lobster shrimp octopus jellyfish starfish coral bat bats hedgehog mole
badger hare hares antelope gazelle reptile reptiles amphibian mammal mammals
primate gorilla chimpanzee orangutan sloth armadillo ferret hamster gerbil
chipmunk weasel mink boar stallion mare colt hound terrier poodle bulldog
retriever spaniel shepherd husky dalmatian chihuahua tabby feline canine bovine
equine livestock herd swarm predator prey wildlife fauna creature creatures beast
beasts monkeys puppies kittens human people 
""".split()

LANDSCAPES = """
land water forest forests tree trees ocean oceans sea seas lake lakes river
rivers stream streams pond ponds mountain mountains hill hills valley valleys
beach beaches shore shoreline coast coastal coastline cliff cliffs desert deserts
meadow meadows field fields grass grassland prairie jungle rainforest woodland
woods swamp swamps marsh wetland wetlands bog waterfall waterfalls cave caves
canyon canyons gorge plateau ridge peak peaks summit glacier glaciers iceberg
snow snowy ice sand sandy rock rocks rocky stone stones boulder pebble mud muddy
dirt soil clay gravel foliage leaf leaves branch branches trunk bark root roots
flower flowers blossom petal petals bush bushes shrub shrubs fern ferns moss vine
vines reed reeds lily lilies cactus palm palms pine pines oak oaks maple maples
willow willows birch cedar redwood bamboo cherry elm spruce cypress sky skies
cloud clouds cloudy sun sunny sunset sunrise sunshine horizon rainbow storm
storms rain rainy fog foggy mist misty dew wind windy lightning thunder twilight
dusk dawn moon moonlight star stars island islands peninsula bay bays lagoon
estuary delta reef reefs tide tides wave waves waters current currents pool pools
puddle spring springs creek creeks brook brooks tundra savanna savannah oasis
dune dunes volcano volcanic terrain landscape landscapes scenery scenic
wilderness nature natural environment outdoors habitat habitats ecosystem
vegetation greenery garden gardens park parks orchard vineyard vineyards farmland
pasture pastures countryside edge edges waterfall waterfalls waterbody water bodies wetlands shorelines 
coastlines coasts seashore seashores riverbank riverbanks bank banks boulders 
plant plants grassland flora underbrush undergrowth canopy tree canopy
grove groves water surface water surfaces ripples ripple floating vegetation 
aquatic plants algae seaweed overcast fog bank skyline 
""".split()

OBJECTS = """
    fence fences gate gates bridge bridges boat boats ship ships canoe kayak sail
    sailboat dock docks pier piers wharf buoy anchor net nets oar oars paddle house
    houses home homes cabin cabins cottage cottages barn barns roof roofs wall walls
    door doors window windows post posts pole poles tower towers building buildings
    hut huts shed sheds lighthouse windmill tent tents road roads path paths trail
    trails bench benches chair chairs table tables umbrella umbrellas basket baskets
    bucket buckets ladder ladders wheel wheels wagon cart carts sign signs flag flags
    rope ropes chain chains hat hats coat coats jacket jackets boot boots glove gloves
    scarf scarves bag bags backpack backpacks camera cameras binoculars telescope
    lamp lamps lantern candle candles bottle bottles cup cups plate plates bowl bowls
    jar jars box boxes crate crates barrel barrels wire wires cable cables pipe pipes
    brick bricks board boards plank log logs stump wood wooden statue statues fountain
    fountains well wells mill mills
""".split()

# Colors, materials, and surface textures — the vocabulary of *backgrounds*
# (land vs. water, forest vs. sky), so highly relevant to spurious-cue concepts.
COLORS_MATERIALS_TEXTURES = """
black white grey gray brown red orange yellow green blue purple pink violet
turquoise teal navy maroon crimson scarlet beige tan cream ivory golden gold
silver bronze copper amber olive khaki charcoal slate rust ruby emerald azure
indigo lavender magenta chestnut auburn brownish greenish bluish reddish
colorful colourful pale bright dark light vivid muted metallic glossy matte shiny
transparent translucent opaque smooth rough coarse glossy furry fuzzy feathered
scaly spotted striped speckled mottled patterned textured wet dry damp muddy dusty
grassy leafy sandy rocky stony icy snowy foggy misty murky reflective glistening
wooden metal metallic iron steel plastic glass stone marble granite concrete
ceramic clay leather cotton wool silk fabric cloth rubber
""".split()

# Bird body parts — fine-grained anatomy for bird concept naming.
BIRD_BODY_PARTS = """
beak bill wing wings feather feathers plumage tail talon talons claw claws crest
breast throat crown belly nape rump flank wingtip wingspan down quill plume neck
""".split()

# Animal body parts.
ANIMAL_BODY_PARTS = """
fur tail paw paws claw claws hoof hooves horn horns antler antlers mane snout
muzzle whiskers fang fangs tusk tusks trunk scales fin fins gills shell hide pelt
ear ears nose tongue teeth leg legs udder hump spine underbelly
""".split()

# Human body parts.
HUMAN_BODY_PARTS = """
face hand hands arm arms leg legs foot feet head hair eye eyes nose mouth lips
ear ears finger fingers chin cheek forehead shoulder shoulders chest neck knee
elbow skin teeth tongue thumb wrist ankle back hip hips waist beard eyebrow
""".split()

CATEGORIES = [
    ("animals",    ANIMALS),
    ("landscapes", LANDSCAPES),
    ("objects",    OBJECTS),
    ("colors_materials_textures", COLORS_MATERIALS_TEXTURES),
    ("bird_body_parts",   BIRD_BODY_PARTS),
    ("animal_body_parts", ANIMAL_BODY_PARTS),
    ("human_body_parts",  HUMAN_BODY_PARTS),
]


def load_cub_species(metadata_path):
    """Return the 200 CUB-200-2011 class names (cleaned, class-index order).

    Read live from the Waterbirds metadata so the list is exactly the classes
    this dataset uses, e.g. "001.Black_footed_Albatross" -> "black footed albatross".
    """
    if not os.path.isfile(metadata_path):
        print(f"  [warn] metadata not found ({metadata_path}); skipping CUB species.")
        return []
    by_index = {}
    with open(metadata_path) as f:
        for row in csv.DictReader(f):
            cls_dir = row["img_filename"].split("/")[0]
            m = re.match(r"(\d+)\.(.+)", cls_dir)
            if m:
                by_index[int(m.group(1))] = m.group(2).replace("_", " ").strip().lower()
    return [by_index[i] for i in sorted(by_index)]


def main():
    with open(SOURCE) as f:
        source_words = [w.strip() for w in f if w.strip()]
    source_set = set(source_words)

    kept = []          # (term, category) in category order, deduped
    seen = set()

    def add(term, category):
        if term and term not in seen:
            seen.add(term)
            kept.append((term, category))

    for name, candidates in CATEGORIES:
        for w in candidates:
            add(w, name)

    cub_species = load_cub_species(METADATA)
    for sp in cub_species:
        add(sp, "cub_species")

    all_categories = [c for c, _ in CATEGORIES] + ["cub_species"]

    with open(OUTPUT, "w") as f:
        for term, _ in kept:
            f.write(term + "\n")

    # ── Report ──────────────────────────────────────────────────────────────
    # DISECT coverage is informational only: the list is re-embedded via CLIP,
    # so terms absent from DISECT are still fully usable.
    in_disect = sum(1 for t, _ in kept if t in source_set)
    print(f"Domain vocabulary : {len(kept)} terms  -> {OUTPUT}")
    print(f"DISECT coverage   : {in_disect}/{len(kept)} terms also in clip_disect_20k "
          f"(rest are embedded fresh via CLIP text encoder)")
    print()
    for name in all_categories:
        terms = [t for t, c in kept if c == name]
        cov = sum(1 for t in terms if t in source_set)
        print(f"  {name:<26}: {len(terms):3d}  (in DISECT: {cov})")


if __name__ == "__main__":
    main()
