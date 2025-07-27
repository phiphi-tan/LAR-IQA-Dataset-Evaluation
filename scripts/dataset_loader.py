from datasets import load_dataset

def tally_qa(split_size, is_complex=None):
    ds = load_dataset("vikhyatk/tallyqa-test", split="test")
    if is_complex is not None:
        if is_complex:
            ds = ds.filter(lambda row: not row["qa"][0]["is_simple"])
        else:
            ds = ds.filter(lambda row: row["qa"][0]["is_simple"])
    
    ds = ds.shuffle(seed=split_size) # for random selection
    input_dataset = ds.select(range(split_size))
    image_list = input_dataset['image']

    return image_list

def text_ocr(split_size):
    ds = load_dataset("MiXaiLL76/TextOCR_OCR", split="train")
    ds = ds.shuffle(seed=split_size) # for random selection
    input_dataset = ds.select(range(split_size))
    image_list = input_dataset['image'] # get list of images

    return image_list

def ocr_vqa(split_size):
    ds = load_dataset("howard-hou/OCR-VQA", split="train")
    ds = ds.shuffle(seed=split_size) # for random selection
    input_dataset = ds.select(range(split_size))
    image_list = input_dataset['image'] # get list of images

    return image_list

def drone_detection(split_size):
    ds = load_dataset("pathikg/drone-detection-dataset", split="test")
    # filter for only single-drone detection
    ds = ds.filter(lambda row: len(row['objects']['category']) == 1)
    ds = ds.shuffle(seed=split_size) # for random selection
    input_dataset = ds.select(range(split_size))
    image_list = input_dataset['image'] # get list of images

    return image_list

def weapon_detection(split_size):
    ds = load_dataset("KIRANKALLA/weaponds", split="train")
    # filter for only single-drone detection
    ds = ds.filter(lambda row: len(row['objects']['categories']) == 1)
    ds = ds.shuffle(seed=split_size) # for random selection
    input_dataset = ds.select(range(split_size))
    image_list = input_dataset['image'] # get list of images

    return image_list

def flickr8k(split_size):
    ds = load_dataset("Naveengo/flickr8k", split="train")
    ds = ds.shuffle(seed=split_size) # for random selection
    input_dataset = ds.select(range(split_size))
    image_list = input_dataset['image'] # get list of images
    
    return image_list
