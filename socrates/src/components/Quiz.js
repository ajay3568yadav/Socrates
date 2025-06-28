import React, { useState } from 'react';
import '../css/Quiz.css';

const Quiz = ({ quizData, onSubmit, isLoading }) => {
  const [userAnswers, setUserAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);

  const handleAnswerSelect = (questionId, answerIndex) => {
    if (submitted) return; // Prevent changes after submission
    
    setUserAnswers(prev => ({
      ...prev,
      [questionId]: answerIndex
    }));
  };

  const handleSubmit = () => {
    // Check if all questions are answered
    const allAnswered = quizData.questions.every(q => 
      userAnswers.hasOwnProperty(q.id)
    );
    
    if (!allAnswered) {
      alert('Please answer all questions before submitting.');
      return;
    }
    
    setSubmitted(true);
    console.log('Submitting quiz with answers:', userAnswers);
    onSubmit(userAnswers);
  };

  const resetQuiz = () => {
    setUserAnswers({});
    setSubmitted(false);
  };

  if (!quizData || !quizData.questions) {
    return null;
  }

  return (
    <div className="quiz-container">
      <div className="quiz-header">
        <div className="quiz-icon">🧠</div>
        <div className="quiz-title">
          <h3>CUDA Programming Quiz</h3>
          <p className="quiz-topic">{quizData.topic}</p>
        </div>
        <div className="quiz-info">
          <span className="question-count">{quizData.questions.length} Questions</span>
        </div>
      </div>

      <div className="quiz-questions">
        {quizData.questions.map((question, index) => (
          <div key={question.id} className="quiz-question">
            <div className="question-header">
              <span className="question-number">Question {index + 1}</span>
              <span className="question-type">{question.type === 'mcq' ? 'Multiple Choice' : 'True/False'}</span>
            </div>
            
            <h4 className="question-text">{question.question}</h4>
            
            <div className="question-options">
              {question.options.map((option, optionIndex) => (
                <label 
                  key={optionIndex} 
                  className={`option-label ${
                    userAnswers[question.id] === optionIndex ? 'selected' : ''
                  } ${submitted ? 'disabled' : ''}`}
                >
                  <input
                    type="radio"
                    name={`question-${question.id}`}
                    value={optionIndex}
                    checked={userAnswers[question.id] === optionIndex}
                    onChange={() => handleAnswerSelect(question.id, optionIndex)}
                    disabled={submitted}
                    className="option-radio"
                  />
                  <span className="option-text">{option}</span>
                  <span className="option-marker"></span>
                </label>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="quiz-actions">
        {!submitted ? (
          <button 
            onClick={handleSubmit}
            disabled={isLoading}
            className="submit-quiz-btn"
          >
            {isLoading ? '⏳ Submitting...' : '✅ Submit Quiz'}
          </button>
        ) : (
          <div className="quiz-submitted">
            <span className="submitted-text">✅ Quiz submitted! Generating feedback...</span>
            <button 
              onClick={resetQuiz}
              className="retake-quiz-btn"
              disabled={isLoading}
            >
              🔄 Retake Quiz
            </button>
          </div>
        )}
        
        <div className="quiz-progress">
          Answered: {Object.keys(userAnswers).length}/{quizData.questions.length}
        </div>
      </div>
    </div>
  );
};

export default Quiz;