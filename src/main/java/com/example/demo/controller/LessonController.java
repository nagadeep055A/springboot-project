package com.example.demo.controller;


import com.example.demo.entity.Lesson;
import com.example.demo.entity.Course;
import com.example.demo.service.LessonService;
import com.example.demo.repository.CourseRepository;

import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/lessons")
@CrossOrigin(origins = "http://localhost:3000")
public class LessonController {

    private final LessonService lessonService;
    private final CourseRepository courseRepository;

    public LessonController(LessonService lessonService, CourseRepository courseRepository) {
        this.lessonService = lessonService;
        this.courseRepository = courseRepository;
    }

    // 1️⃣ Add lesson to a course
    @PostMapping("/add/{courseId}")
    public Lesson addLesson(@PathVariable Long courseId, @RequestBody Lesson lesson) {

        Course course = courseRepository.findById(courseId)
                .orElseThrow(() -> new RuntimeException("Course not found"));

        lesson.setCourse(course); // attach lesson to course
        return lessonService.saveLesson(lesson);
    }

    // 2️⃣ Get all lessons for a course
    @GetMapping("/course/{courseId}")
    public List<Lesson> getLessons(@PathVariable Long courseId) {
        return lessonService.getLessonsByCourseId(courseId);
    }
}
