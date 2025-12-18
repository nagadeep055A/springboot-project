
package com.example.demo.entity;

import jakarta.persistence.*;
import java.util.*;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonManagedReference;

@Entity
@Table(name = "courses")
@JsonIgnoreProperties({"hibernateLazyInitializer", "handler"})

public class Course {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String title;        // Example: Core Java, Spring Boot, React.js
    private String category;     // Example: "backend", "frontend", "database", "fullstack"

    @Lob
    @Column(columnDefinition = "TEXT")
    private String description;  // Full detailed course info

    // One Course → Many Lessons
    @OneToMany(mappedBy = "course", cascade = CascadeType.ALL, orphanRemoval = true)
    @JsonManagedReference
    private List<Lesson> lessons = new ArrayList<>();

    public Course() {}

    public Course(String title, String category, String description) {
        this.title = title;
        this.category = category;
        this.description = description;
    }

    // ------------------ Getters & Setters ------------------

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getTitle() { return title; }
    public void setTitle(String title) { this.title = title; }

    public String getCategory() { return category; }
    public void setCategory(String category) { this.category = category; }

    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }

    public List<Lesson> getLessons() { return lessons; }
    public void setLessons(List<Lesson> lessons) { 
        this.lessons = lessons; 
    }

    // Helper method to add Lesson to Course
    public void addLesson(Lesson lesson) {
        lesson.setCourse(this);
        this.lessons.add(lesson);
    }

	@Override
	public String toString() {
		return "Course [id=" + id + ", title=" + title + ", category=" + category + ", description=" + description
				+ ", lessons=" + lessons + "]";
	}

//	public String getName() {
//		// TODO Auto-generated method stub
//		return null;
//	}
//    
  
}





