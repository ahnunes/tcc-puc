# -*- coding: utf-8 -*-

"""
Store Scrapy items into CSV file
https://docs.scrapy.org/en/latest/topics/item-pipeline.html
"""

import csv
import os.path
import re


class ScraperPipeline(object):
    """ Pipeline definition """

    file = None
    writer = None

    def open_spider(self, spider): #pylint: disable=unused-argument
        """ Create output.csv with CSV header """
        filename = 'output.csv'
        exists = os.path.exists(filename)

        self.file = open(filename, 'a+', encoding='utf-8')
        self.writer = csv.writer(self.file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        if not exists:
            self.writer.writerow(['source', 'title', 'text', 'rating'])
            self.file.flush()

    def close_spider(self, spider): #pylint: disable=unused-argument
        """ Close file """
        self.file.flush()
        self.file.close()

    def normalize(self, text):
        """ Remove double space, line breaks and trim the text """
        if text is None:
            return ""

        return re.sub(r'  +', ' ', re.sub(r'[\n\r]', ' ', text)).strip()

    def process_item(self, item, spider):
        """ Write scrapy item into CSV file """
        self.writer.writerow([
            spider.name,
            self.normalize(item['title']),
            self.normalize(item['text']),
            float(item['rating'])
        ])

        return item
