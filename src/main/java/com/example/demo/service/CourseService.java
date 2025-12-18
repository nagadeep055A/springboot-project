package com.example.demo.service;

import java.util.List;
import java.util.Optional;

import com.example.demo.entity.Course;
import com.example.demo.entity.CourseDTO;

public interface CourseService {

    List<Course> getAllCourses();

    List<Course> getCoursesByCategory(String category);

    Optional<Course> getCourseById(Long id);

    // ⭐ New method to return course WITH lessons
    Course getCourseWithLessons(Long id);

    Course saveCourse(Course course);

    void deleteCourse(Long id);
//    CourseDTO getCourseBasic(Long id);

    
   

}
