import { useState } from 'react';
import FlashcardList from './FlashcardList';
import FLASHCARD_DATA from './data';

function App() {
     const [flashcards, setFlashcards] = useState(FLASHCARD_DATA)
  return (
    <>
      <FlashcardList flashcards={flashcards} />
    </>
  );
}

export default App