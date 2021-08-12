import os 

''' Load the API key '''
key = os.environ.get('AZURE_SEARCH_KEY', 'XXX')

results = search_images_bing(key, 'grizzly bear')
ims = results.attrgot('content_url')
len(ims)

dest = 'images/grizzly.jpg'
download_url(ims[0], dest)

# Formatting to universal image size
im = Image.open(dest)
im.to_thumb(128, 128)

# sorting the bears and putting them in folders
bear_types = 'grizzly', 'black', 'teddy'
path = Path('bears')

if not path.exists():
    path.mkdir()
    for o in bear_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(key,f'{o} bear')
        download_images(deset, urls=results.attrgot('content_url'))

# verify any broken files 
failed = verify_images(fns)
print(failed)

# remove any failed images by unlinking
failed.map(Path.unlink);

''' Create a data loader '''
class DataLoaders(GetAttr):
    def __init__(self, *loaders): self.loaders= loaders
    def __getitem__(self, i): return self.loaders[i]
    train, valid = add_props(lambda i, self: self[i])

bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42)
    get_y = parent_label,
    item_tfms=Resize(128)
)

dls = bears.dataloaders(path)
dls.valid.show_bath(max_n=4, nrows=1)

# pull and squish them 
bears = bears.new(item_tfms=Resize(128, ResizeMethod.Squish))
dls = bears.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)

# pad  the images 
bears = bears.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'))
dls = bears.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)

# random resize and crop
bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
dls = bears.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1, unique=True)


''' Data Augmentation '''
bears = bears.net(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)

bears = bears.new(
    item_tfms = RandomResizedCrop(224, min_scale=0.5),
    batch_tfms = aug_transforms()
)
dls = bears.dataloaders(path)

# create the model learner
learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tuen(4)

