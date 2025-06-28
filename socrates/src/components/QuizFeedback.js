import React from 'react';
import '../css/QuizFeedback.css';

const QuizFeedback = ({ evaluation }) => {
  if (!evaluation) return null;

  const getScoreColor = (percentage) => {
    if (percentage >= 80) return '#22c55e'; // Green
    if (percentage >= 60) return '#f97316'; // Orange  
    return '#ef4444'; // Red
  };

  const getScoreIcon = (percentage) => {
    if (percentage >= 80) return '🎉';
    if (percentage >= 60) return '👍';
    return '📚';
  };

  return (
    <div className="quiz-feedback-container">
      <div className="feedback-header">
        <div className="score-display">
          <div 
            className="score-circle"
            style={{ borderColor: getScoreColor(evaluation.percentage) }}
          >
            <span className="score-icon">{getScoreIcon(evaluation.percentage)}</span>
            <span className="score-text">
              {evaluation.score}/{evaluation.total}
            </span>
            <span className="score-percentage">
              {evaluation.percentage.toFixed(0)}%
            </span>
          </div>
        </div>
        
        <div className="feedback-summary">
          <h3>Quiz Results</h3>
          <p className="overall-message">{evaluation.overall_message}</p>
        </div>
      </div>

      <div className="detailed-feedback">
        <h4>Question by Question Review</h4>
        
        {evaluation.detailed_feedback.map((feedback, index) => (
          <div 
            key={feedback.question_id} 
            className={`feedback-item ${feedback.is_correct ? 'correct' : 'incorrect'}`}
          >
            <div className="feedback-question">
              <span className="question-number">Q{index + 1}</span>
              <span className="feedback-icon">
                {feedback.is_correct ? '✅' : '❌'}
              </span>
              <p className="question-text">{feedback.question}</p>
            </div>
            
            <div className="feedback-answers">
              <div className="answer-row">
                <span className="answer-label">Your answer:</span>
                <span className={`answer-text ${feedback.is_correct ? 'correct' : 'incorrect'}`}>
                  {feedback.user_answer_text}
                </span>
              </div>
              
              {!feedback.is_correct && (
                <div className="answer-row">
                  <span className="answer-label">Correct answer:</span>
                  <span className="answer-text correct">
                    {feedback.correct_answer_text}
                  </span>
                </div>
              )}
            </div>
            
            <div className="feedback-explanation">
              <span className="explanation-label">💡 Explanation:</span>
              <p className="explanation-text">{feedback.explanation}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="feedback-actions">
        <div className="study-suggestions">
          <h4>📖 Keep Learning</h4>
          <p>Want to practice more? Try asking:</p>
          <ul>
            <li>"Give me a quiz about CUDA memory types"</li>
            <li>"Quiz me on CUDA kernel optimization"</li>
            <li>"Test my knowledge of CUDA threading"</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default QuizFeedback;