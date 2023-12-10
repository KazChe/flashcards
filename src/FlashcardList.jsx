import PropTypes from 'prop-types';
import Flashcard from "./Flashcard"

export default function FlashcardList({ flashcards }) {
    return(
        <div className="card-grid">
            {flashcards.map(flashcard => {
                return <Flashcard flashcard={flashcard} key={flashcard.id} />
            })}
        </div>
    )
}

FlashcardList.propTypes = {
    flashcards: PropTypes.array.isRequired
};
