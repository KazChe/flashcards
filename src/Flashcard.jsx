import { useState } from "react"
import PropTypes from "prop-types"

export default function Flashcard({ flashcard }) {
    const [flip, setFlip] = useState(false)
    let theObject = {__html:flashcard.answer}
    // let optionsObject = {__html:flashcard.options}
    return(
        <div className={`card ${flip ? 'flip' : ''}`} onClick={() => setFlip(!flip)}>
            {flip ? <div dangerouslySetInnerHTML={theObject} /> : flashcard.question +' '+ flashcard.options } 
       </div>
    )
}

Flashcard.propTypes = {
    flashcard: PropTypes.object.isRequired
}