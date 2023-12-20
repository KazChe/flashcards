import { useState } from 'react';
import FlashcardList from './FlashcardList';
import FLASHCARD_DATA from './data';
import FlashcardInputForm from './FalshcardInputForm';

function App() {
     const [flashcards, setFlashcards] = useState(FLASHCARD_DATA)
  return (
    <>
      {/* <FlashcardInputForm /><br/> */}
      <FlashcardList flashcards={flashcards} />
    </>
  );
}

export default App