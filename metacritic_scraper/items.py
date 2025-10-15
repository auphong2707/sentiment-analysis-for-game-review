# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class GameReviewItem(scrapy.Item):
    """Item for storing individual game review data"""
    
    # Game information
    game_title = scrapy.Field()
    game_platform = scrapy.Field()
    game_url = scrapy.Field()
    game_metascore = scrapy.Field()
    game_release_date = scrapy.Field()
    game_genre = scrapy.Field()
    game_developer = scrapy.Field()
    game_publisher = scrapy.Field()
    
    # Review information
    review_id = scrapy.Field()
    review_text = scrapy.Field()
    review_score = scrapy.Field()
    review_category = scrapy.Field()  # positive, mixed, or negative
    reviewer_name = scrapy.Field()
    review_date = scrapy.Field()
    review_url = scrapy.Field()
    
    # Additional metadata
    scraped_at = scrapy.Field()


class GameItem(scrapy.Item):
    """Item for storing game metadata"""
    
    game_title = scrapy.Field()
    game_platform = scrapy.Field()
    game_url = scrapy.Field()
    game_metascore = scrapy.Field()
    game_user_score = scrapy.Field()
    game_release_date = scrapy.Field()
    game_genre = scrapy.Field()
    game_developer = scrapy.Field()
    game_publisher = scrapy.Field()
    game_rating = scrapy.Field()
    game_summary = scrapy.Field()
    total_critic_reviews = scrapy.Field()
    total_user_reviews = scrapy.Field()
    scraped_at = scrapy.Field()
