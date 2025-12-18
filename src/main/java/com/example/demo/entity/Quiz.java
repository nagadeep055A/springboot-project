//package com.example.demo.entity;
//
//import jakarta.persistence.*;
//import java.util.*;
//
//@Entity
//@Table(name = "quizzes")
//public class Quiz {
//
//    @Id
//    @GeneratedValue(strategy = GenerationType.IDENTITY)
//    private Long id;
//
//    private String title;
//    private String description;
//
//    // QUIZ BELONGS TO A LESSON
//    @ManyToOne(fetch = FetchType.LAZY)
//    @JoinColumn(name = "lesson_id")
//    private Lesson lesson;
//
//    // QUIZ HAS MANY QUESTIONS
//    @OneToMany(mappedBy = "quiz", cascade = CascadeType.ALL, orphanRemoval = true)
//    private List<Question> questions = new ArrayList<>();
//
//    public Quiz() {}
//
//    // Getters & Setters
//    public Long getId() { return id; }
//    public String getTitle() { return title; }
//    public String getDescription() { return description; }
//    public Lesson getLesson() { return lesson; }
//    public List<Question> getQuestions() { return questions; }
//
//    public void setTitle(String title) { this.title = title; }
//    public void setDescription(String description) { this.description = description; }
//    public void setLesson(Lesson lesson) { this.lesson = lesson; }
//
//    public void addQuestion(Question q) {
//        questions.add(q);
//        q.setQuiz(this);
//    }
//}
