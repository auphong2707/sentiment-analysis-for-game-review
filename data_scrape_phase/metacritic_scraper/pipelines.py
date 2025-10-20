# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import csv
import json
import os
from datetime import datetime


class MetacriticScraperPipeline:
    """Basic pipeline for cleaning and processing items"""
    
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        
        # Clean text fields
        if adapter.get('review_text'):
            adapter['review_text'] = adapter['review_text'].strip()
        
        # Convert scores to integers
        if adapter.get('review_score'):
            try:
                adapter['review_score'] = int(adapter['review_score'])
            except (ValueError, TypeError):
                adapter['review_score'] = None
        
        # Ensure category is set
        if adapter.get('review_score') and not adapter.get('review_category'):
            score = adapter['review_score']
            if score >= 75:
                adapter['review_category'] = 'positive'
            elif score >= 50:
                adapter['review_category'] = 'mixed'
            else:
                adapter['review_category'] = 'negative'
        
        return item


class CsvExportPipeline:
    """Pipeline to export items to CSV"""
    
    def open_spider(self, spider):
        """Initialize CSV file when spider opens"""
        os.makedirs('data', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.file_path = f'data/metacritic_reviews_{timestamp}.csv'
        self.file = open(self.file_path, 'w', newline='', encoding='utf-8')
        self.writer = None
        spider.logger.info(f'Saving reviews to: {self.file_path}')
    
    def close_spider(self, spider):
        """Close CSV file when spider closes"""
        self.file.close()
        spider.logger.info(f'Reviews saved to: {self.file_path}')
    
    def process_item(self, item, spider):
        """Write item to CSV"""
        adapter = ItemAdapter(item)
        
        # Initialize CSV writer with headers on first item
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=adapter.field_names())
            self.writer.writeheader()
        
        self.writer.writerow(adapter.asdict())
        return item


class JsonExportPipeline:
    """Pipeline to export items to JSON"""
    
    def open_spider(self, spider):
        """Initialize JSON file when spider opens"""
        os.makedirs('data', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.file_path = f'data/metacritic_reviews_{timestamp}.json'
        self.file = open(self.file_path, 'w', encoding='utf-8')
        self.items = []
        spider.logger.info(f'Saving reviews to: {self.file_path}')
    
    def close_spider(self, spider):
        """Write all items to JSON file when spider closes"""
        json.dump(self.items, self.file, indent=2, ensure_ascii=False)
        self.file.close()
        spider.logger.info(f'Reviews saved to: {self.file_path}')
    
    def process_item(self, item, spider):
        """Add item to list"""
        adapter = ItemAdapter(item)
        self.items.append(dict(adapter))
        return item


class DuplicatesPipeline:
    """Pipeline to filter duplicate reviews"""
    
    def __init__(self):
        self.ids_seen = set()
    
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        review_id = adapter.get('review_id')
        
        if review_id:
            if review_id in self.ids_seen:
                spider.logger.debug(f'Duplicate review found: {review_id}')
                raise DropItem(f'Duplicate item found: {review_id}')
            else:
                self.ids_seen.add(review_id)
        
        return item


from scrapy.exceptions import DropItem
