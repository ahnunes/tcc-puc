# -*- coding: utf-8 -*-

"""
Scrapy definition for Magazine Luiza - https://www.magazineluiza.com.br/
"""

import json

import scrapy
from scraper.items import ScraperItem


class Magalu(scrapy.Spider):
    """ Spider definition for Magazine Luiza """

    name = 'Magalu'
    categories = [
        "smartphone/celulares-e-smartphones/s/te/tcsp",  # Celulares
        "tvs/tv-e-video/s/et/tves",  # TVs
        "notebook/informatica/s/in/note",  # Notebook
        "maquina-de-lavar/eletrodomesticos/s/ed/lava",  # Máquina de Lavar
        "sofas/moveis/s/mo/msof",  # Sofas
        "perfume/beleza-e-perfumaria/s/pf/pftm",  # Perfume
    ]

    def start_requests(self):
        """ Generate initial requests """
        for category in self.categories:
            for page in range(50): # Load 50 pages
                url = self.search_url(category, page)
                yield scrapy.Request(url)

    def search_url(self, category, page):
        """ Return search URL """
        return 'https://www.magazineluiza.com.br/{}?sort=type%3AsoldQuantity%2Corientation%3Adesc&page={}'.format(
            category, page + 1)

    def review_url(self, product_id, page):
        """ Return review URL """
        if len(product_id) == 7:
            product_id += "00"

        return 'https://www.magazineluiza.com.br/review/{}/?page={}'.format(product_id, page)

    def parse_reviews(self, response):
        """ Parse review page """
        try:
            data = json.loads(response.text)['data']
        except Exception:  # pylint: disable=broad-except
            return

        for review in data['objects']:
            yield ScraperItem(
                title=review['title'],
                text=review['review_text'],
                rating=review['rating'],
            )

        if data['pages'] > response.meta['page']:
            product_id = response.meta['id']
            next_page = response.meta['page'] + 1
            url = self.review_url(id, next_page)

            yield scrapy.Request(url, callback=self.parse_reviews, meta={'page': next_page, 'id': product_id})

    def parse(self, response):
        """ Parse search page """
        scripts = response.css('script[type="application/ld+json"]')

        for script in scripts:
            if script is not None:
                content = script.css('::text').get()

                if "sku" in content:
                    try:
                        product_id = json.loads(content)['sku']
                        url = self.review_url(id, 1)
                        yield scrapy.Request(url, callback=self.parse_reviews, meta={'page': 1, 'id': product_id})
                    except Exception:  # pylint: disable=broad-except
                        continue
