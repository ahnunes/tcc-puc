# -*- coding: utf-8 -*-

"""
Define scrapy item
https://docs.scrapy.org/en/latest/topics/items.html
"""

import scrapy


class ScraperItem(scrapy.Item):
    """ Define field list """
    title = scrapy.Field()
    text = scrapy.Field()
    rating = scrapy.Field()
