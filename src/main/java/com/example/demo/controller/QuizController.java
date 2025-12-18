
package com.example.demo.controller;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.example.demo.entity.Course;
import com.example.demo.entity.Option;
import com.example.demo.entity.Question;
import com.example.demo.repository.CourseRepository;
import com.example.demo.repository.OptionRepository;
import com.example.demo.repository.QuestionRepository;

@RestController
@RequestMapping("/api/quiz")
@CrossOrigin("*")
public class QuizController {

    @Autowired
    private CourseRepository courseRepository;

    @Autowired
    private QuestionRepository questionRepository;

    @Autowired
    private OptionRepository optionRepository;

    // ------------------------------------------------
    // 1️⃣ Add question (you already have)
    // ------------------------------------------------
    @PostMapping("/add/{courseId}")
    public String addQuestion(@PathVariable Long courseId, @RequestBody Question question) {
        Course course = courseRepository.findById(courseId)
                .orElseThrow(() -> new RuntimeException("Course not found"));

        question.setCourse(course);
        Question savedQ = questionRepository.save(question);

        for (Option opt : question.getOptions()) {
            opt.setQuestion(savedQ);
            optionRepository.save(opt);
        }

        return "Question added!";
    }

    // ------------------------------------------------
    // 2️⃣ Get quiz questions for a course
    // ------------------------------------------------
    @GetMapping("/course/{courseId}")
    public List<Question> getCourseQuiz(@PathVariable Long courseId) {
        return questionRepository.findByCourseId(courseId);
    }

    // ------------------------------------------------
    // 3️⃣ Submit answers and calculate score
    // ------------------------------------------------
    @PostMapping("/submit")
    public int submitQuiz(@RequestBody List<Long> selectedOptionIds) {

        int score = 0;

        for (Long optionId : selectedOptionIds) {
            Option opt = optionRepository.findById(optionId)
                    .orElse(null);

            if (opt != null && opt.isCorrect()) {
                score++;
            }
        }

        return score;
    }
}
