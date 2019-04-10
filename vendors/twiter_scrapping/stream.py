from __future__ import absolute_import, print_function

import sys

from tqdm import tqdm
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener

# Go to http://apps.twitter.com and create an app.
# The consumer key and secret will be generated for you after
consumer_key = "66DVuBOEAFFsIOr4YvH9gmbUX"
consumer_secret = "Th7t2xet2DwChViWV5pjVw5ms4wNjWUWHbbbGeVoROQmkRMY5H"

# After the step above, you will be redirected to your app's page.
# Create an access token under the the "Your access token" section
access_token = "3409482433-LVECR4YPwyc3JMnuedAG2RAm67u1R4Vfk0KuMd4"
access_token_secret = "EEIw0siFiTU1x7XT6fypPyYn51J7VMVC0f1yzBPIHUYgM"


class StdOutListener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.

    """

    def __init__(self, path=None, index=0):
        self.folder = path
        self.index = index
        self.pbar = tqdm(initial=index)
        super().__init__()

    def on_data(self, data):
        with open(self.folder + "/" + str(self.index) + ".json", "w") as  f:
            self.index += 1
            f.write(data)
            self.pbar.update(1)

        return True

    def on_error(self, status):
        print(status)


if __name__ == '__main__':
    l = StdOutListener("./stream_data/" + sys.argv[1], int(sys.argv[2]))
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    stream = Stream(auth, l)
    stream.sample(languages=[sys.argv[1]], is_async=True)
