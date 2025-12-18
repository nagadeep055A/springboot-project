package com.example.demo.service;

import java.util.List;
import com.example.demo.entity.Option;

public interface OptionService {
    List<Option> getOptionsByQuestion(Long questionId);
}
