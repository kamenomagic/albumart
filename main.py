#! /usr/bin/python
from spotify import Spotify


def main():
    spotify = Spotify()
    results = spotify.test_search('Queen')
    for result in results:
        print(result)


if __name__ == '__main__':
    main()
