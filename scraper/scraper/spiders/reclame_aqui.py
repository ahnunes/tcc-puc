# -*- coding: utf-8 -*-

"""
Scrapy definition for Reclame Aqui - https://www.reclameaqui.com.br/
"""

import json

import scrapy
from scraper.items import ScraperItem


class ReclameAqui(scrapy.Spider):
    """ Spider definition for Reclame Aqui """

    name = 'ReclameAqui'
    categories = [
        "780",    # Correios
        "8383",   # Netshoes
        "105",    # Caixa
        "4421",   # Vivo
        "87803",  # Uber
        "91961",  # C&A
        "12552",  # Decolar
        "902",    # NET
        "19773",  # Mercado Pago
    ]

    def start_requests(self):
        """ Generate initial requests """
        for category in self.categories:
            for page in range(300):
                url = self.search_url(category, page)
                yield scrapy.Request(url)

    def search_url(self, category, page):
        """ Return search URL """
        return 'https://iosearch.reclameaqui.com.br/raichu-io-site-search-v1/query/companyComplains/10/{}?company={}&evaluated=bool:true'.format( # pylint: disable=line-too-long
            page * 10, category)

    def parse(self, response):
        """ Parse items from search page """
        data = json.loads(response.text)['complainResult']['complains']['data']

        for review in data:
            yield ScraperItem(
                title=review['evaluation'],
                text=review['description'],
                rating=review['score'],
            )
