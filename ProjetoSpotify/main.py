import csv
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Set up Spotify credentials
client_id = '2306216f0fbd41928b696062a0115e31'
client_secret = '2471b33d51194b209d38344985afa087'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Function to search for tracks based on artist name
def search_tracks_by_artist(artist_name):
    results = sp.search(q=f'artist:{artist_name}', type='track', limit=10)
    return results['tracks']['items']

# Function to get track features
def get_track_features(track_id):
    track_features = sp.audio_features(track_id)
    return track_features[0] if track_features else None

# Function to save data to CSV
def save_to_csv(data, filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, filename)
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

# Main function
def main():
    artist_name = 'Frejat'
    tracks = search_tracks_by_artist(artist_name)

    # Extract track data and IDs
    track_data = []
    for track in tracks:
        track_id = track['id']
        print(track_id)
        track_title = track['name']
        artist_name = track['artists'][0]['name']
        track_features = get_track_features(track_id)
        if track_features:
            track_data.append({
                'artist': artist_name,
                'track': track_title,
                'id': track_id,
                'danceability': track_features['danceability'],
                'energy': track_features['energy'],
                'key': track_features['key'],
                'loudness': track_features['loudness'],
                'mode': track_features['mode'],
                'speechiness': track_features['speechiness'],
                'acousticness': track_features['acousticness'],
                'instrumentalness': track_features['instrumentalness'],
                'liveness': track_features['liveness'],
                'valence': track_features['valence'],
                'tempo': track_features['tempo'],
                'duration_ms': track_features['duration_ms'],
                'time_signature': track_features['time_signature']
            })
    
    # Save data to CSV
    save_to_csv(track_data, f'{artist_name}_track_features.csv')

if __name__ == "__main__":
    main()
