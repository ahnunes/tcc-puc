# -*- coding: utf-8 -*-

"""
Scrapy definition for B2W / Americanas - https://www.americanas.com.br/
"""

import json
import urllib.parse

import scrapy
from scraper.items import ScraperItem


class B2W(scrapy.Spider):
    """ Spider definition for B2W / Americanas """

    name = 'B2W'
    categories = [
        "229187",  # Celulares
        "228926",  # Beleza e Perfumaria
        "227644",  # Eletrodomésticos
        "444385",  # Sofá
        "267868",  # Notebook
    ]

    def start_requests(self):
        """ Generate initial requests """
        for category in self.categories:
            for page in range(10):
                url = self.search_url(category, page)
                yield scrapy.Request(url)

    def search_url(self, category, page):
        """ Return search URL """
        offset = page * 100
        filter_param = urllib.parse.quote(
            '{{"id":"category.id","value":"{}","hidden":false,"fixed":true}}'.format(category))
        return 'https://mystique-v2-americanas.juno.b2w.io/search?offset={}&sortBy=topSelling&source=omega&filter={}&limit=100'.format( # pylint: disable=line-too-long
            offset, filter_param)

    def review_url(self, product_id, page):
        """ Return review URL """
        offset = (page - 1) * 5
        return 'https://product-reviews-bff-v1-americanas.b2w.io/reviews?&offset={}&limit=5&sort=SubmissionTime:desc&filter=ProductId:{}'.format( # pylint: disable=line-too-long
            offset, product_id)

    def parse_reviews(self, response):
        """ Parse review page """
        data = json.loads(response.text)

        if 'Results' not in data:
            return

        for review in data['Results']:
            yield ScraperItem(
                title=review['Title'],
                text=review['ReviewText'],
                rating=review['Rating'],
            )

        if data['TotalResults'] > response.meta['page'] * 5:
            product_id = response.meta['id']
            next_page = response.meta['page'] + 1
            url = self.review_url(id, next_page)

            yield scrapy.Request(url, callback=self.parse_reviews, meta={'page': next_page, 'id': product_id})

    def parse(self, response):
        """ Parse search page """
        data = json.loads(response.text)

        for product in data['products']:
            product_id = product['id']
            url = self.review_url(product_id, 1)
            yield scrapy.Request(url, callback=self.parse_reviews, meta={'page': 1, 'id': product_id})
