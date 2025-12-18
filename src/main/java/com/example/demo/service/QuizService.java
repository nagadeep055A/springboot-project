package com.example.demo.service;

import java.util.List;

public interface QuizService {
    String addQuestion(Long courseId, Object questionData);
    List<?> getQuizByCourse(Long courseId);
    int submitQuiz(List<Long> selectedOptionIds);
}
