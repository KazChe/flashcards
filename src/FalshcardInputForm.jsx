import { useState } from "react"

export default function FlashcardInputForm() {
    const [formData, setFormData] = useState({question: "", answer: ""})

        const handleChange = (event) => {
            const {name, value} = event.target
            setFormData((prevFormData) => ({...prevFormData, [name]: value}))
        }

        const handleSubmit = (event) => {
            event.preventDefault()
            alert(`question: ${formData.question}, answer: ${formData.answer}`)
        }
    
        return (
            <form onSubmit={handleSubmit}>
                <label htmlFor="question">Question:</label>
                <input type="textarea" name="question" value={formData.question} onChange={handleChange}/>
                <label htmlFor="answer">Answer:</label>
                <textarea name="answer" value={formData.answer} onChange={handleChange}/>
                <button type="submit">Submit</button>
            </form>
        )
    }