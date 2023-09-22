import time
import genanki

def create_anki_deck(deck_name, question_list, answer_list, output_file="output.apkg"):
    # With model as a template
    model_id = int(time.time()) 
    my_model = genanki.Model(model_id, 'Simple Model',
                            fields=[
                                {'name': 'Question'},
                                {'name': 'Answer'},
                            ],
                            templates=[
                                {
                                'name': 'Card 1',
                                'qfmt': '{{Question}}',
                                'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}',
                                },
                            ])
    
    deck_id = int(time.time()) 
    # Create deck with notes
    my_deck = genanki.Deck(deck_id, deck_name)

    # With notes with model
    for tmp_question, tmp_answer in zip(question_list, answer_list):
        my_note = genanki.Note(model=my_model, fields=[tmp_question, tmp_answer])
        my_deck.add_note(my_note)

    # Save
    genanki.Package(my_deck).write_to_file(output_file)

if __name__ == '__main__':
    deck_name = "Test deck"
    question_list = ['What is 1+1?', 'Who is Danny?', 'Who is Haydn?']
    answer_list = ['42', 'Danny is thinking.', 'Haydn is running.']

    create_anki_deck(deck_name, question_list, answer_list, output_file="output.apkg")

