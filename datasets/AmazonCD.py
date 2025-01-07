#!/usr/bin/env python3

from datasets.Amazon import Amazon


class AmazonCD(Amazon):
    DATASETURL = 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/CDs_and_Vinyl.json.gz'
    REMOTE_FILENAME = 'CDs_and_Vinyl.json.gz'

    METADATAURL = 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_CDs_and_Vinyl.json.gz'
    METADATA_FILENAME = 'meta_CDs_and_Vinyl.json.gz'

    CAT_TO_DROP = ['CDs & Vinyl']
