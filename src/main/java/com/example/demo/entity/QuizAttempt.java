package com.example.demo.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
public class QuizAttempt {

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private Long quizId;
    private String userIdentifier; // optional (email or session id) - can be null
    private Integer score;
    private LocalDateTime attemptedAt = LocalDateTime.now();

    public QuizAttempt(){}

    public QuizAttempt(Long quizId, String userIdentifier, Integer score){
        this.quizId = quizId; this.userIdentifier = userIdentifier; this.score = score;
    }

    // getters setters
    public Long getId(){return id;}
    public void setId(Long id){this.id=id;}
    public Long getQuizId(){return quizId;}
    public void setQuizId(Long quizId){this.quizId=quizId;}
    public String getUserIdentifier(){return userIdentifier;}
    public void setUserIdentifier(String userIdentifier){this.userIdentifier=userIdentifier;}
    public Integer getScore(){return score;}
    public void setScore(Integer score){this.score=score;}
    public LocalDateTime getAttemptedAt(){return attemptedAt;}
    public void setAttemptedAt(LocalDateTime attemptedAt){this.attemptedAt=attemptedAt;}
}

