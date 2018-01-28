import spotipy
client_id = '784e52cf923944efb42b9bc46191bcf3'
client_secret = '6cae2c711e4d4e03bac22ed3b21e618b'
spotify = spotipy.Spotify()

results = spotify.search(q='weezer', limit=20)
for i, t in enumerate(results['tracks']['items']):
    print(' ', i, t['name'])