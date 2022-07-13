from app_utils import *

if __name__ == '__main__':
    genre_model = createGenreContinuationModel()
    predictNwGenreModel(genre_model, './inp_guitar_piano_chords_3.mid',\
    genre = 'jazz', temperature_notes = 1.5, temperature_duration = 1.0,\
    mem_len = 2048)

    