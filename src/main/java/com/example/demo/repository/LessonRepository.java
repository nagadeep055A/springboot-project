package com.example.demo.repository;

import com.example.demo.entity.Lesson;
import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

public interface LessonRepository extends JpaRepository<Lesson, Long> {

    // Get all lessons for a particular course
    List<Lesson> findByCourseId(Long courseId);
}
