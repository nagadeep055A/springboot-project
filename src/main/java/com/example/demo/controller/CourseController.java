package com.example.demo.controller;

import com.example.demo.entity.Course;
import com.example.demo.entity.CourseDTO; // Imported but unused
import com.example.demo.service.CourseService;
import com.example.demo.service.LessonService;

import jakarta.servlet.http.HttpSession; // Imported but unused

import java.util.List;

import org.springframework.http.ResponseEntity; // Imported but unused
import org.springframework.ui.Model; // Used in viewCourseDetails
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping; // Imported but unused
import org.springframework.web.bind.annotation.RestController;

/**
 * Controller for handling HTTP requests related to Course entities.
 * The @CrossOrigin annotation allows requests from the React frontend running on port 3000.
 */
@CrossOrigin(origins = "http://localhost:3000")
@RestController
public class CourseController {

    private final CourseService courseService;
    private final LessonService lessonService;

    /**
     * Constructor for Dependency Injection (Spring injects services).
     */
    public CourseController(CourseService courseService, LessonService lessonService) {
        this.courseService = courseService;
        this.lessonService = lessonService;
    }

    /**
     * Endpoint to retrieve a list of all courses.
     * Maps to: GET /courses
     * Returns JSON data for the React frontend.
     */
    @GetMapping("/courses")
    public List<Course> getCourses() {
        return courseService.getAllCourses(); // ✔ sends JSON to React
    }

    /**
     * Endpoint used for traditional Server-Side Rendering (e.g., using Thymeleaf or JSP).
     * Maps to: GET /courses/{id}
     * This method adds data to the Model for view rendering.
     * * NOTE: This endpoint returns a view name "course-details", not JSON.
     */
    @GetMapping("/courses/{id}")
    public String viewCourseDetails(@PathVariable Long id, Model model) {
        // Fetches data and adds it to the model object
        model.addAttribute("course", courseService.getCourseById(id));
        model.addAttribute("lessons", lessonService.getLessonsByCourseId(id));
        return "course-details"; // View name to resolve
    }
    
    /**
     * Dedicated REST API endpoint for the React frontend to fetch full course details.
     * Maps to: GET /api/courses/{id}
     * Returns JSON data, including the associated lessons (handled by the service layer).
     */
    @GetMapping("/api/courses/{id}")
    public Course getCourseById(@PathVariable Long id) {
        // Uses the service method that fetches the Course and populates its Lesson list
        return courseService.getCourseWithLessons(id);
    }
    
    @PostMapping("/api/courses/{id}")
    public Course addCourse(@RequestBody Course course) {
        return courseService.saveCourse(course);
    }

    
}