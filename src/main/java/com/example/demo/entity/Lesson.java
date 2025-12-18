package com.example.demo.entity;

import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonBackReference;
import com.fasterxml.jackson.annotation.JsonManagedReference;

import jakarta.persistence.*;

@Entity
public class Lesson {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String title;

    @Lob
    private String theory;

    @Lob
    private String codeSample;

    // ✅ Many Lessons -> Belong to One Course
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "course_id")
    @JsonBackReference
    private Course course;

    // ✅ One Lesson -> Many Quizzes
//    @OneToMany(mappedBy = "lesson", cascade = CascadeType.ALL, orphanRemoval = true)
//    @JsonManagedReference
//    private List<Quiz> quizzes = new ArrayList<>();

    public Lesson() {}

    public Lesson(String title, String theory, String codeSample) {
        this.title = title;
        this.theory = theory;
        this.codeSample = codeSample;
    }

    // -------- Getters & Setters --------

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getTitle() { return title; }
    public void setTitle(String title) { this.title = title; }

    public String getTheory() { return theory; }
    public void setTheory(String theory) { this.theory = theory; }

    public String getCodeSample() { return codeSample; }
    public void setCodeSample(String codeSample) { this.codeSample = codeSample; }

    public Course getCourse() { return course; }
    public void setCourse(Course course) { this.course = course; }

//    public List<Quiz> getQuizzes() { return quizzes; }
//    public void setQuizzes(List<Quiz> quizzes) { this.quizzes = quizzes; }
//
//    // Helper method
//    public void addQuiz(Quiz quiz) {
//        quiz.setLesson(this);
//        this.quizzes.add(quiz);
//    }
}
