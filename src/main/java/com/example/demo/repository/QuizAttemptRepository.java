package com.example.demo.repository;

import com.example.demo.entity.QuizAttempt;
import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

public interface QuizAttemptRepository extends JpaRepository<QuizAttempt, Long> {
    List<QuizAttempt> findByQuizIdOrderByAttemptedAtDesc(Long quizId);
}
