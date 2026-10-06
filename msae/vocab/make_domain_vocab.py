"""Build a visually-grounded domain vocabulary for SAE concept naming.

One curated term list per dataset, selected with --dataset:

  waterbirds  birds, animals, landscapes/nature, objects, colours/materials/
              textures and body parts, plus the 200 CUB-200-2011 species read
              live from the Waterbirds metadata.csv.
  celeba      faces and facial features, hair (colour, length, style), people
              and gender words, accessories and makeup, expressions, portrait
              photography, plus the 40 CelebA attribute names read live from
              list_attr_celeba.

The vocabulary is consumed by re-embedding it through CLIP's text encoder
(msae/precompute_activations.py), NOT by reusing the fixed DISECT embeddings --
so words do NOT need to appear in clip_disect_20k.txt. We therefore keep every
curated term, and only *report* DISECT coverage for information. The CUB
species and the CelebA attribute names in particular are largely absent from
DISECT, which is exactly why they are worth adding.

The per-dataset "live" terms are read from the dataset itself rather than
hand-transcribed, so they are exactly the classes/attributes in use.

Output is named from the dataset's DatasetSpec.concept_vocab, so it matches
what the pipeline asks for with --concept_match_vocab.

Usage:
    python msae/vocab/make_domain_vocab.py                      # waterbirds
    python msae/vocab/make_domain_vocab.py --dataset celeba
    python msae/vocab/make_domain_vocab.py --dataset celeba --data_dir /path/to/data
"""

import argparse
import csv
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SOURCE = os.path.join(SCRIPT_DIR, "clip_disect_20k.txt")
sys.path.insert(0, REPO_ROOT)

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

# ── CelebA candidate terms ──────────────────────────────────────────────────
# The target is hair colour and the spurious attribute is gender, so the list
# leans on hair, faces and person words; accessories / expressions / portrait
# terms cover what else a CelebA crop actually contains.
#
# One term PER LINE, not whitespace-split like the waterbirds lists above:
# most of what matters here is multi-word ("blond hair", "receding hairline",
# "wearing lipstick"), and splitting on spaces would shred exactly the phrases
# the naming step needs.

def _lines(block):
    """One term per line, blank lines and '#' comments ignored."""
    out = []
    for line in block.strip().split("\n"):
        term = line.split("#")[0].strip().lower()
        if term:
            out.append(term)
    return out


HAIR = _lines("""
hair
hairstyle
haircut
hairline
receding hairline
widow's peak
scalp
blond
blonde
blond hair
blonde hair
golden hair
platinum blonde
light hair
dark hair
black hair
brown hair
brunette
chestnut hair
auburn hair
red hair
ginger hair
grey hair
gray hair
silver hair
white hair
bald
balding
bald head
shaved head
buzz cut
crew cut
bangs
fringe
curly hair
wavy hair
straight hair
frizzy hair
long hair
short hair
shoulder length hair
ponytail
pigtails
hair bun
braid
braided hair
cornrows
dreadlocks
afro
updo
bob cut
pixie cut
hair highlights
dyed hair
hair roots
hair parting
side part
centre part
slicked back hair
messy hair
tousled hair
shiny hair
thick hair
thin hair
hair colour
blond eyebrows
dark eyebrows
""")

FACES = _lines("""
face
facial features
portrait
head
profile
headshot
eye
eyes
eyebrow
eyebrows
arched eyebrows
bushy eyebrows
eyelashes
eyelid
narrow eyes
wide eyes
nose
pointy nose
big nose
nostrils
mouth
lips
big lips
thin lips
teeth
tongue
mouth slightly open
cheek
cheeks
high cheekbones
rosy cheeks
chin
double chin
jaw
jawline
forehead
brow
ear
ears
neck
throat
skin
complexion
pale skin
fair skin
tan skin
olive skin
dark skin
wrinkles
crease
dimples
freckles
mole
beauty mark
blemish
scar
stubble
5 o clock shadow
beard
no beard
moustache
goatee
sideburns
facial hair
clean shaven
oval face
round face
chubby face
bags under eyes
""")

PEOPLE = _lines("""
person
people
man
men
woman
women
male
female
boy
girl
gentleman
lady
adult
child
teenager
young man
young woman
elderly person
middle aged person
masculine face
feminine face
celebrity
actor
actress
model
singer
performer
public figure
crowd
couple
""")

ACCESSORIES_MAKEUP = _lines("""
glasses
eyeglasses
sunglasses
reading glasses
hat
cap
beanie
beret
headband
bandana
scarf
veil
hood
earring
wearing earrings
necklace
wearing necklace
pendant
choker
jewelry
piercing
nose ring
necktie
wearing necktie
bow tie
collar
shirt
suit
jacket
dress
makeup
heavy makeup
lipstick
wearing lipstick
lip gloss
eyeliner
eyeshadow
mascara
foundation makeup
blush
nail polish
""")

EXPRESSIONS = _lines("""
smiling
smile
laughing
grin
frowning
serious expression
neutral expression
mouth open
mouth closed
eyes open
eyes closed
squinting
winking
raised eyebrows
furrowed brow
surprised
happy expression
sad expression
angry expression
looking at camera
looking away
head tilt
attractive
""")

PORTRAIT_PHOTO = _lines("""
photograph
photo
snapshot
selfie
close up
red carpet
press event
premiere
camera flash
studio lighting
soft lighting
harsh lighting
backlight
shadow
highlight
blurred background
bokeh
plain background
dark background
light background
indoor
outdoor
microphone
spotlight
stage
banner
logo
watermark
grainy
blurry
sharp focus
""")

SKIN_COLOURS = _lines("""
black
white
brown
blond
golden
silver
grey
gray
red
auburn
light
dark
""")

WATERBIRDS_CATEGORIES = [
    ("animals",    ANIMALS),
    ("landscapes", LANDSCAPES),
    ("objects",    OBJECTS),
    ("colors_materials_textures", COLORS_MATERIALS_TEXTURES),
    ("bird_body_parts",   BIRD_BODY_PARTS),
    ("animal_body_parts", ANIMAL_BODY_PARTS),
    ("human_body_parts",  HUMAN_BODY_PARTS),
]
# NOTE: BIRDS is defined above but has never been part of this list, so none
# of its exclusive terms (penguin, albatross, toucan, aviary, ...) are in the
# published waterbirds vocabulary. Left as-is on purpose: the concept_match
# .npy files were built from the current list, and adding terms would shift
# every concept name. Add ("birds", BIRDS) here and regenerate the .npy if
# you do want them.

CELEBA_CATEGORIES = [
    ("hair",                HAIR),
    ("faces",               FACES),
    ("people",              PEOPLE),
    ("accessories_makeup",  ACCESSORIES_MAKEUP),
    ("expressions",         EXPRESSIONS),
    ("portrait_photo",      PORTRAIT_PHOTO),
    ("skin_colours",        SKIN_COLOURS),
    ("colors_materials_textures", COLORS_MATERIALS_TEXTURES),
    ("objects",             OBJECTS),
    ("human_body_parts",    HUMAN_BODY_PARTS),
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


def load_celeba_attributes(data_dir):
    """The 40 CelebA attribute names, read live from list_attr_celeba so the
    list is exactly what this copy of the dataset annotates, e.g.
    "5_o_Clock_Shadow" -> "5 o clock shadow", "Wearing_Lipstick" ->
    "wearing lipstick". The CelebA analogue of the CUB species list.
    """
    root = os.path.join(data_dir, "celeba")
    csv_p, txt_p = (os.path.join(root, "list_attr_celeba.csv"),
                    os.path.join(root, "list_attr_celeba.txt"))
    header = None
    if os.path.isfile(csv_p):
        with open(csv_p) as f:
            header = f.readline().strip().split(",")[1:]
    elif os.path.isfile(txt_p):
        with open(txt_p) as f:
            f.readline()                       # image count
            header = f.readline().split()
    if not header:
        print(f"  [warn] list_attr_celeba not found under {root}; "
              f"skipping CelebA attribute names.")
        return []
    return [re.sub(r"[_\s]+", " ", a).strip().lower() for a in header if a]


# name -> (category table, live-term loader, description of the live terms)
DATASETS = {
    "waterbirds": (WATERBIRDS_CATEGORIES,
                   lambda d: load_cub_species(os.path.join(d, "waterbirds", "metadata.csv")),
                   "cub_species"),
    "celeba":     (CELEBA_CATEGORIES, load_celeba_attributes, "celeba_attributes"),
}


def build(dataset, data_dir, output=None):
    """Write <dataset>'s vocabulary and return (path, kept terms)."""
    import dataset_settings
    if dataset not in DATASETS:
        raise SystemExit(f"No term lists for {dataset!r}; known: {sorted(DATASETS)}. "
                         f"Add a CATEGORIES table and a live-term loader above.")
    categories, live_loader, live_name = DATASETS[dataset]
    vocab_name = dataset_settings.get(dataset).concept_vocab or f"{dataset}_domain"
    output = output or os.path.join(SCRIPT_DIR, f"{vocab_name}_vocab.txt")

    with open(SOURCE) as f:
        source_set = {w.strip() for w in f if w.strip()}

    kept, seen = [], set()

    def add(term, category):
        if term and term not in seen:
            seen.add(term)
            kept.append((term, category))

    for name, candidates in categories:
        for w in candidates:
            add(w, name)
    for term in live_loader(data_dir):
        add(term, live_name)

    with open(output, "w") as f:
        for term, _ in kept:
            f.write(term + "\n")

    # ── Report ──────────────────────────────────────────────────────────────
    # DISECT coverage is informational only: the list is re-embedded via CLIP,
    # so terms absent from DISECT are still fully usable.
    in_disect = sum(1 for t, _ in kept if t in source_set)
    print(f"Dataset           : {dataset}  (vocab name: {vocab_name})")
    print(f"Domain vocabulary : {len(kept)} terms  -> {output}")
    print(f"DISECT coverage   : {in_disect}/{len(kept)} terms also in clip_disect_20k "
          f"(rest are embedded fresh via CLIP text encoder)")
    print()
    for name in [c for c, _ in categories] + [live_name]:
        terms = [t for t, c in kept if c == name]
        cov = sum(1 for t in terms if t in source_set)
        print(f"  {name:<28}: {len(terms):3d}  (in DISECT: {cov})")
    return output, kept


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="waterbirds", choices=sorted(DATASETS))
    ap.add_argument("--data_dir", default=os.path.join(REPO_ROOT, "data"),
                    help="parent of the dataset folder, for the live-term lists")
    ap.add_argument("--output", default=None,
                    help="output path (default: msae/vocab/<concept_vocab>_vocab.txt)")
    args = ap.parse_args()
    build(args.dataset, args.data_dir, args.output)


if __name__ == "__main__":
    main()
