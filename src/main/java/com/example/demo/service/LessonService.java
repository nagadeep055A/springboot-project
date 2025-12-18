package com.example.demo.service;

import com.example.demo.entity.Lesson;
import java.util.List;
import java.util.Optional;

public interface LessonService {

    List<Lesson> getLessonsByCourseId(Long courseId);

    Optional<Lesson> getLessonById(Long id);

    Lesson saveLesson(Lesson lesson);

    void deleteLesson(Long id);
}

