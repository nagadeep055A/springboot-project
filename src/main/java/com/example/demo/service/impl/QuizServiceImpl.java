package com.example.demo.service.impl;

import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;

import java.util.List;

import com.example.demo.entity.*;
import com.example.demo.repository.*;
import com.example.demo.service.QuizService;

@Service
public class QuizServiceImpl implements QuizService {

    @Autowired private CourseRepository courseRepository;
    @Autowired private QuestionRepository questionRepository;
    @Autowired private OptionRepository optionRepository;

    @Override
    public String addQuestion(Long courseId, Object body) {
        // (Not implementing — unnecessary because controller handles JSON directly)
        return null;
    }

    @Override
    public List<Question> getQuizByCourse(Long courseId) {
        return questionRepository.findByCourseId(courseId);
    }

    @Override
    public int submitQuiz(List<Long> optionIds) {
        int score = 0;

        for (Long id : optionIds) {
            Option opt = optionRepository.findById(id).orElse(null);
            if (opt != null && opt.isCorrect()) score++;
        }
        return score;
    }
}
