package com.example.demo.service.impl;

import java.util.List;
import java.util.Optional;

import org.springframework.stereotype.Service;

import com.example.demo.entity.CourseDTO; // Note: Not used in implemented methods, only commented out code
import com.example.demo.entity.Course;
import com.example.demo.entity.Lesson;
import com.example.demo.repository.CourseRepository;
import com.example.demo.repository.LessonRepository;
import com.example.demo.service.CourseService;

/**
 * Implementation of the CourseService interface.
 * Uses Constructor Injection for the required repositories.
 */
@Service
public class CourseServiceImpl implements CourseService {

    private final CourseRepository courseRepository;
    private final LessonRepository lessonRepository;

    /**
     * Constructor for Dependency Injection.
     * Spring automatically injects the repository implementations.
     */
    public CourseServiceImpl(CourseRepository courseRepository, LessonRepository lessonRepository) {
        this.courseRepository = courseRepository;
        this.lessonRepository = lessonRepository;
    }

    // --- Standard CRUD Methods ---

    @Override
    public List<Course> getAllCourses() {
        return courseRepository.findAll();
    }

    @Override
    public Optional<Course> getCourseById(Long id) {
        return courseRepository.findById(id);
    }

    @Override
    public Course saveCourse(Course course) {
        return courseRepository.save(course);
    }

    @Override
    public void deleteCourse(Long id) {
        courseRepository.deleteById(id);
    }
    
    // --- Custom Query Methods ---

    
    // ⭐⭐ NEW METHOD — React needs this to load full course details 
    /**
     * Retrieves a Course entity by ID and manually fetches and attaches its associated Lessons.
     * Throws a RuntimeException if the Course is not found.
     */
    @Override 
    public Course getCourseWithLessons(Long id) {
        // 1. Fetch Course and throw exception if Optional is empty
        Course course = courseRepository.findById(id)
            .orElseThrow(() -> new RuntimeException("Course not found with ID: " + id));

        // 2. Fetch related Lessons using the dedicated LessonRepository method
        // Assumes LessonRepository has a method defined like:
        // List<Lesson> findByCourseId(Long courseId);
        List<Lesson> lessons = lessonRepository.findByCourseId(id);

        // 3. Attach the List of Lessons to the Course object (mutator/setter method required in Course entity)
        course.setLessons(lessons);
        
        return course;
    }

	@Override
	public List<Course> getCoursesByCategory(String category) {
		// TODO Auto-generated method stub
		return null;
	}

    // public CourseDTO getCourseBasic(Long id) { // ... (Commented out DTO projection method)
    // }
}