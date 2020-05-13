import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'first_project.settings')

import django

django.setup()

import random
from first_app.models import Topic, Webpage, AccessRecord
from faker import Faker


# Fake POP Script
fakeGen = Faker()
topics = ['Search', 'Social', 'Market Place', 'News', 'Games']


def add_topic():
    t = Topic.objects.get_or_create(top_name=random.choice(topics))[0]
    t.save()
    return t


def populate(n=5):
    for entry in range(n):
        top = add_topic()
        fake_url = fakeGen.url()
        fake_date = fakeGen.date()
        fake_name = fakeGen.company()

        web = Webpage.objects.get_or_create(topic=top, url=fake_url, name=fake_name)[0]

        acc_rec = AccessRecord.objects.get_or_create(name=web, date=fake_date)[0]


if __name__ == '__main__':
    print('populating script!')
    populate(20)
    print('Populating complete!')
