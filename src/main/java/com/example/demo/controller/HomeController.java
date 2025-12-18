package com.example.demo.controller;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@CrossOrigin(origins="https//localhost:3000")

@RestController
public class HomeController {

    @GetMapping("/homes")
    public String homePage() {
        return "/header.jsp";
    }

}
