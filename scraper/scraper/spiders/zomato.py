# -*- coding: utf-8 -*-

"""
Scrapy definition for Zomato - https://www.zomato.com/pt/
"""

import json

import scrapy
from scraper.items import ScraperItem


class Zomato(scrapy.Spider):
    """ Spider definition for Zomato """

    name = 'Zomato'
    categories = [
        "sao-paulo-sp",
        "rio",
        "salvador",
        "portoalegre",
        "brasilia",
    ]

    def start_requests(self):
        """ Generate initial requests """
        for category in self.categories:
            for page in range(50):
                url = self.search_url(category, page)
                yield scrapy.Request(url)

    def search_url(self, category, page):
        """ Return search URL """
        return 'https://www.zomato.com/pt/{}/restaurantes?page={}'.format(category, page + 1)

    def review_url(self, product_id, page):
        """ Return review URL """
        return 'https://www.zomato.com/webroutes/reviews/loadMore?sort=dd&filter=reviews-dd&res_id={}&page={}'.format(
            product_id, page)

    def parse_reviews(self, response):
        """ Parse review page """
        data = json.loads(response.text)

        if ('entities' not in data) or ('RATING' not in data['entities']):
            return

        ratings = data['entities']['RATING']

        for review in data['entities']['REVIEWS'].values():
            yield ScraperItem(
                title="",
                text=review['reviewText'],
                rating=ratings[str(review['rating']['entities'][0]['entity_ids'][0])]['rating'],
                source="Zomato",
            )

        if data['page_data']['sections']['SECTION_REVIEWS']['numberOfPages'] > response.meta['page']:
            product_id = response.meta['id']
            next_page = response.meta['page'] + 1
            url = self.review_url(id, product_id)

            yield scrapy.Request(url, callback=self.parse_reviews, meta={'page': next_page, 'id': product_id})

    def parse(self, response):
        product_ids = response.css('.js-search-result-li::attr(data-res_id)').getall()

        for product_id in product_ids:
            try:
                url = self.review_url(product_id, 1)
                yield scrapy.Request(url, callback=self.parse_reviews, meta={'id': product_id, 'page': 1})
            except Exception: # pylint: disable=broad-except
                continue
