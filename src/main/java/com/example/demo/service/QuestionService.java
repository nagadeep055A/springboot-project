package com.example.demo.service;

import java.util.List;
import com.example.demo.entity.Question;

public interface QuestionService {
    List<Question> getQuestionsByCourse(Long courseId);
}
