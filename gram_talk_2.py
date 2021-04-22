from manim import *
import numpy as np
import itertools as it
from colour import Color

def rainbowify(textmobj):
    colors_of_the_rainbow = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE] 
    colors = it.cycle(colors_of_the_rainbow)
    for letter in textmobj:
        letter.set_color(next(colors))


class WhatAboutNonParallelotopes(Scene):
    def construct(self):
        all_text = Tex(r"Gramian works ", r"for parallelotopes").scale(1.5)
        all_text[1].set_color(RED)
        
        but_text = Tex(r"What about a general shape?", color=YELLOW).scale(1.5).next_to(all_text, DOWN)
        
        self.play(Write(all_text))
        self.wait()
        self.play(Write(but_text))
        self.wait()

      
class SphereExample(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(z_min=-4.5, z_max=4.5)
        self.set_camera_orientation(phi=60 * DEGREES, theta=-100 * DEGREES)
        
        self.add(axes)
        
        unit_sphere = Sphere()
        
        self.play(Create(unit_sphere))
        self.wait()
        
        apply_text = Tex(r"Apply some nonlinear transformation into 4D").to_edge(UP).add_background_rectangle()
        
        self.add_fixed_in_frame_mobjects(apply_text)
        self.play(Write(apply_text[1]))
        self.wait()
        
        question_text = Tex(r"What is the new volume?", color=YELLOW).next_to(apply_text, DOWN).add_background_rectangle()
        
        self.add_fixed_in_frame_mobjects(question_text)
        self.play(Write(question_text[1]))
        self.wait()
        
        would_text = Tex(r"Would need some sort of ", r"integral...").next_to(question_text, DOWN).add_background_rectangle()
        would_text[2].set_color(GREEN)
        self.add_fixed_in_frame_mobjects(would_text)
        self.play(Write(would_text[1:]))
        self.wait()
        
        question_text_2 = Tex(r"How to integrate over a 3D region in 4D?", color=YELLOW).next_to(would_text, DOWN).add_background_rectangle()
        self.add_fixed_in_frame_mobjects(question_text_2)
        self.play(Write(question_text_2[1]))
        self.wait()


class ThePlan0(Scene):
    def construct(self):
        title_text = Tex(r"The Plan").scale(2).to_edge(UP)
        
        self.play(Write(title_text))
        self.wait()
        
        line_1 = Tex(r"Recap ", r"integration", r" and ", r"change of variables", r" in 1D").shift(UP)
        line_1[1].set_color(YELLOW)
        line_1[3].set_color(YELLOW)
        self.play(Write(line_1))
        self.wait()
        
        line_2 = Tex(r"See how to extend this to ", r"2D area integrals")
        line_2[1].set_color(YELLOW)
        self.play(Write(line_2))
        self.wait()
        
        line_3 = Tex(r"See how this extends to ", r"(generalised) surface integrals").shift(DOWN)
        line_3[1].set_color(YELLOW)
        self.play(Write(line_3))
        self.wait()
        
        box = SurroundingRectangle(line_1)
        self.play(Create(box))
        self.wait()

### Adding in this part for the "fake-out"

class HangOnAMinute(Scene):
    def construct(self):
        hold_text_1 = Tex(r"Actually, hold on...").scale(2.5).shift(UP)
        hold_text_2 = Tex(r"Before the maths...", color=YELLOW).scale(2.5).next_to(hold_text_1, DOWN)
        
        self.play(Write(hold_text_1))
        self.wait()
        self.play(Write(hold_text_2))
        self.wait()

class MightNoticeIveBeenRecapping(Scene):
    def construct(self):
        might_text_1 = Tex(r"Might have noticed I've been").scale(1.6).shift(UP)
        might_text_2 = Tex(r"recapping previous concepts", r" a lot").scale(1.6).next_to(might_text_1, DOWN)
        might_text_2[0].set_color(YELLOW)
        
        self.play(Write(might_text_1))
        self.play(Write(might_text_2))
        self.wait()

class MathsIsPrettyCumulative(Scene):
    def construct(self):
        title_text = Tex(r"Maths is ", r"Cumulative").scale(2.5).to_edge(UP)
        title_text[1].set_color(YELLOW)
        
        self.play(Write(title_text))
        self.wait()
        
        rects = list()
        
        for count in range(15):
            temp_rect = Rectangle(height=0.4, width=4, fill_opacity=1, fill_color=GREEN, stroke_color=WHITE, stroke_width=1)
            if rects:
                temp_rect.align_to(rects[count - 1], UP).shift(UP*0.4)
            else:
                temp_rect.to_edge(DOWN)
            rects.append(temp_rect)
                
        
        rects = VGroup(*rects)
        rects.set_fill(opacity=0)
        self.play(Create(rects))
        self.wait()
        
        
        for count in range(7):
            self.play(rects[count].animate.set_fill(opacity=1), run_time=0.4)
            self.wait(0.2)
        
        rects[7].set_fill(color=GREY)
        self.play(rects[7].animate.set_fill(opacity=1), run_time=0.5)
        self.wait()
        
        self.play(FadeOutAndShift(rects[7], LEFT))
        for count in range(8, 15):
            rects[count].set_fill(color=RED)
            self.play(rects[count].animate.set_fill(opacity=1), run_time=0.1)
        self.wait()
        
        rects[7].set_fill(color=GREEN)
        self.play(FadeIn(rects[7]))
        self.wait()
        
        for count in range(8, 15):
            self.play(rects[count].animate.set_fill(color=GREEN), run_time=0.1)
        
        self.wait()

class TheresOnlySoMuchICanDo(Scene):
    def construct(self):
        uh_oh_1 = Tex(r"There's only so much").scale(2).shift(UP)
        uh_oh_2 = Tex(r"I can do!").scale(2).next_to(uh_oh_1, DOWN)
        
        self.play(Write(uh_oh_1))
        self.play(Write(uh_oh_2))
        self.wait()
        
        uh_oh_grp = VGroup(uh_oh_1, uh_oh_2)
        
        self.play(uh_oh_grp.animate.to_edge(UP))
        self.wait()
        
        as_text = Tex(r"As \textbf{learners},", color=YELLOW).scale(2)
        as_text_2 = Tex(r"what can we do?", color=YELLOW).scale(2).next_to(as_text, DOWN)
        
        self.play(Write(as_text))
        self.play(Write(as_text_2))
        self.wait()



class ChangeOfVariables1D(GraphScene):
    def __init__(self, **kwargs):
        GraphScene.__init__(
            self,
            exclude_zero_label=False,
            x_min=0,
            x_max=1,
            x_axis_config={"tick_frequency": 0.1},
            x_labeled_nums=[0, 1],
            y_min=0,
            y_max=1.5,
            y_axis_config={"tick_frequency": 0.1},
            y_labeled_nums=[0,1],
            **kwargs
        )
    
    @staticmethod
    def test_func(x):
        return 1 + 50 * x * (x - 0.1) * (x - 0.4) * (x - 0.7) * (x - 1)
        
    @staticmethod
    def test_func_u(u):
        return ChangeOfVariables1D.test_func(2*u)
    
    def construct(self):
        self.setup_axes()
        graph = self.get_graph(ChangeOfVariables1D.test_func, x_min=0, x_max=1)
        self.play(Create(graph))
        self.wait()
        
        label = self.get_graph_label(graph)
        self.play(Write(label))
        self.wait()
        
        integral = MathTex(r"\int_0^1 ", r"f(x)", r"dx", r" \approx \sum_1^n ", r"f(x_k)", "\Delta x")
        integral[1].set_color(BLUE)
        integral[4].set_color(BLUE)
        integral.to_edge(UP)
        
        self.play(Write(integral[:3]))
        
        self.wait()
        
        area_under = self.get_riemann_rectangles(graph, x_min=0, x_max=1, dx=0.01, start_color=GREEN, end_color=BLUE, stroke_width = 0, fill_opacity=1, stroke_color=GREEN)
        
        self.play(Create(area_under))
        
        self.wait()
        
        self.play(FadeOut(area_under))
        
        self.wait()
        
        better_approxs = self.get_riemann_rectangles_list(
            graph,
            7,
            max_dx=0.25,
            x_min=0,
            x_max=1,
            input_sample_type="left",
            start_color=GREEN,
            end_color=BLUE
        )
        
        self.play(Write(integral[3:]))
        self.wait()
        previous_group = False
        for group in better_approxs:
            if previous_group:
                self.play(ReplacementTransform(previous_group, group))
                self.wait(0.2)
            else:
                self.play(Create(group))
                self.wait(0.2)
            previous_group = group
        
        
        self.play(FadeOut(group), FadeOut(integral[3:]))
        
        self.wait()
        
        sub_text = MathTex(r"x", r"=", r"2u")
        sub_text[0].set_color(BLUE)
        sub_text[2].set_color(GREEN)
        sub_text.next_to(integral, RIGHT)
        
        self.play(Write(sub_text))
        self.wait()
        
        u_graph = self.get_graph(ChangeOfVariables1D.test_func_u, x_min=0, x_max=0.5)
        u_label = self.get_graph_label(u_graph, label='f(2u)').shift(DOWN*2)
        
        graph.save_state()
        label.save_state()
        
        self.play(Transform(graph, u_graph), Transform(label, u_label))
        self.wait()
        
        """
        u_label_2 = self.get_graph_label(u_graph, label='f(x(u))').shift(DOWN*2)
        
        self.play(Transform(label, u_label_2))
        """
        
        new_integral = MathTex(r"\int_0^1 ", r"f(x)", r"dx", r" = \int_0^{\frac 12} ", r"?", "du")
        new_integral[1].set_color(BLUE)
        new_integral[4].set_color(GREEN)
        new_integral.to_edge(UP)
        
        self.play(Transform(integral[:3], new_integral))
        self.wait()
        
        particular_approx = self.get_riemann_rectangles(u_graph, x_min=0, x_max=0.5, dx=0.1, start_color=GREEN, end_color=BLUE)
        
        width_brace = Brace(particular_approx[0])
        brace_tex = width_brace.get_tex("\Delta u")
        
        x_approx = self.get_riemann_rectangles(graph, x_min=0, x_max=1, dx=0.2, start_color=GREEN, end_color=BLUE)
        x_width_brace = Brace(x_approx[0])
        x_brace_tex = x_width_brace.get_tex("\Delta x")
        
        self.wait()
        
        approx_group = VGroup(particular_approx, width_brace, brace_tex)
        x_approx_group = VGroup(x_approx, x_width_brace, x_brace_tex)
        
        approx_group.save_state()
        
        self.play(Write(particular_approx))
        self.wait()
        
        self.play(Create(width_brace), Create(brace_tex))
        self.wait()
        
        self.play(Restore(graph), Restore(label), Transform(approx_group, x_approx_group))
        self.wait()
        
        self.wait(2)



class ChangeOfVariables1DDecomposingEffect(Scene):
    def construct(self):
        top_text = MathTex(r"\int_0^1", r"f(x)", r"dx")
        top_text[0].set_color(PURPLE)
        top_text[1].set_color(BLUE)
        top_text[2].set_color(YELLOW)
        
        top_text.to_edge(UP).scale(1.5)
        
        self.play(Write(top_text))
        self.wait()
        
        bottom_mid = (top_text.get_corner(DL) + top_text.get_corner(DR))/2
        
        domain = MathTex(r"\int_0^1", color=PURPLE)
        domain.to_edge(LEFT).shift(RIGHT)
        integrand = MathTex(r"f(x)", color=BLUE)
        differential = MathTex(r"dx", color=YELLOW)
        differential.to_edge(RIGHT).shift(LEFT)
        
        domain_arrow = Arrow(bottom_mid, domain.get_corner(UR), color=PURPLE)
        integrand_arrow = Arrow(bottom_mid, (integrand.get_corner(UR) + integrand.get_corner(UL))/2, color=BLUE)
        differential_arrow = Arrow(bottom_mid, differential.get_corner(UL), color=YELLOW)
        
        domain_brace = Brace(domain, direction=DOWN)
        domain_brace_text = domain_brace.get_text("Domain")
        domain_brace_text.set_color(PURPLE)
        
        integrand_brace = Brace(integrand, direction=DOWN)
        integrand_brace_text = integrand_brace.get_text("Integrand")
        integrand_brace_text.set_color(BLUE)
        
        differential_brace = Brace(differential, direction=DOWN)
        differential_brace_text = differential_brace.get_text("Width element")
        differential_brace_text.set_color(YELLOW)
        
        self.play(Create(domain_arrow), Write(domain))
        self.wait()
        self.play(Create(domain_brace), Create(domain_brace_text))
        self.wait()
        
        self.play(Create(integrand_arrow), Write(integrand))
        self.wait()
        self.play(Create(integrand_brace), Create(integrand_brace_text))
        self.wait()
        
        self.play(Create(differential_arrow), Write(differential))
        self.wait()
        self.play(Create(differential_brace), Create(differential_brace_text))
        self.wait()
        
        sub_text = MathTex(r"x", r"=", r"2u").scale(1.5)
        sub_text[0].set_color(BLUE)
        sub_text[2].set_color(GREEN)
        sub_text.next_to(top_text, RIGHT).to_edge(RIGHT)
        
        self.play(Write(sub_text))
        self.wait()
        
        sub_text_rect = SurroundingRectangle(sub_text, buff=0.1)
        self.play(Create(sub_text_rect))
        self.wait()
        
        new_domain = MathTex(r"\int_0^{\frac 12}", color=PURPLE)
        new_domain.to_edge(LEFT).shift(RIGHT)
        new_domain_rect = SurroundingRectangle(new_domain, buff=0.1)
        new_integrand = MathTex(r"f(2u)", color=GREEN)
        new_integrand_rect = SurroundingRectangle(new_integrand, buff=0.1)
        new_differential = MathTex(r"?", color=YELLOW)
        new_differential.move_to(differential.get_center())
        new_differential_rect = SurroundingRectangle(new_differential, buff=0.2)
        
        self.play(ReplacementTransform(sub_text_rect, new_domain_rect))
        self.wait()
        self.play(Transform(domain, new_domain))
        self.wait()
        self.play(ReplacementTransform(new_domain_rect, new_integrand_rect))
        self.wait()
        self.play(Transform(integrand, new_integrand))
        self.wait()
        self.play(ReplacementTransform(new_integrand_rect, new_differential_rect))
        self.wait()
        self.play(Transform(differential, new_differential))
        self.wait()
        
        know_text = Tex(r"Know how ", r"domain", r" and ", r"integrand", r" change").next_to(integrand_brace_text, DOWN).shift(DOWN*0.5)
        know_text[1].set_color(PURPLE)
        know_text[3].set_color(GREEN)
        
        self.play(Write(know_text))
        self.wait()
        
        harder_text = Tex(r"Need some thinking to find how ", r"width element", r" changes").next_to(know_text, DOWN)
        harder_text[1].set_color(YELLOW)
        
        self.play(Write(harder_text))
        self.wait()



class HowWidthElementTransforms1D(ZoomedScene):
    
    def get_dot_mapping_animation(self, dot_group, func, input_line, output_line):
        animations = list()
        for dot in dot_group:
            new_coords = output_line.n2p(func(input_line.p2n(dot.get_center())))
            animations.append(dot.animate.move_to(new_coords))
        return animations
    
    def zoom_in_on_input(self, input_num, input_line, animate=False):
        zoomed_camera = self.zoomed_camera
        if animate:
            self.play(zoomed_camera.frame.animate.move_to(input_line.n2p(input_num)))
        else:
            zoomed_camera.frame.move_to(input_line.n2p(input_num))
    
    def construct(self):
        note_text = Tex(r"Write ", r"$x$ ", r"$=$ ", r"$g($", r"$u$", r"$)$ ", r"$\longrightarrow$ mapping from ", r"$u$", r" to ", r"$x$")
        note_text.set_color_by_tex("$x$", BLUE)
        note_text.set_color_by_tex("u$", GREEN)
        note_text.to_edge(UP)
        
        fit_text = Tex(r"Construct ", r"$u$", r"-integral that's ", r"equivalent", r" to ", r"$x$", r"-integral")
        fit_text.set_color_by_tex("$x$", BLUE)
        fit_text.set_color_by_tex("u$", GREEN)
        fit_text[3].set_color(YELLOW)
        fit_text.next_to(note_text, DOWN)
        
        specific_func_text_1 = MathTex(r"2u", color=GREEN)
        specific_func_text_1.move_to(note_text[3:6].get_center())
        
        domain_text = MathTex(r"[0, 1] \text{ in }", r"x\text{-world }", r"\longrightarrow", r"[0, \frac 12] \text{ in }", r"u\text{-world}")
        domain_text[1].set_color(BLUE)
        domain_text[-1].set_color(GREEN)
        domain_text.next_to(fit_text, DOWN)
        
        apply_brace = Brace(domain_text[2], DOWN)
        inverse_brace_text = apply_brace.get_text(r"Apply $g^{-1}($", r"$x$", r"$) =$ ", r"$\frac{1}{2} x$ ", r"$=$ ", r"$u$")
        inverse_brace_text.set_color_by_tex("x$", BLUE)
        inverse_brace_text.set_color_by_tex("u$", GREEN)
        inverse_brace_text_simple = apply_brace.get_tex(r"g^{-1}")
        
        normal_brace_text = apply_brace.get_tex(r"g")
        
        
        
        self.play(Write(note_text[:6]))
        self.wait()
        self.play(Write(note_text[6:]))
        self.wait()
        self.play(Write(fit_text))
        self.wait()
        note_text.save_state()
        self.play(Transform(note_text[3:6], specific_func_text_1))
        self.wait()
        self.play(Write(domain_text))
        self.wait()
        self.play(Create(apply_brace), Write(inverse_brace_text))
        self.wait()
        self.play(ReplacementTransform(inverse_brace_text, inverse_brace_text_simple))
        self.wait()
        self.play(FadeOut(apply_brace), FadeOut(inverse_brace_text_simple))
        self.play(CyclicReplace(domain_text[:2], domain_text[3:]))
        self.wait()
        self.play(FadeOut(domain_text[3:]))
        self.wait()
        self.play(Create(apply_brace), Write(normal_brace_text))
        self.wait()
        
        question_brace = Brace(domain_text[3:], DOWN)
        question_brace_text = question_brace.get_text(r"Equivalent domain?")
        question_brace_text.set_color(YELLOW)
        self.play(Create(question_brace), Write(question_brace_text))
        self.wait()
        
        self.play(
            FadeOut(note_text),
            FadeOut(domain_text[:3]),
            FadeOut(question_brace),
            FadeOut(question_brace_text),
            FadeOut(apply_brace),
            FadeOut(normal_brace_text),
            FadeOut(fit_text)
        )
        self.wait()
        
        transform_text = MathTex(r"x", r"=", r"2u").scale(1.5)
        transform_text[0].set_color(BLUE)
        transform_text[2].set_color(GREEN)
        transform_text.to_edge(UP)
        
        self.play(Write(transform_text))
        self.wait()
        
        u_line = NumberLine(
            unit_size=12,
            x_min=0,
            x_max=0.5,
            tick_frequency=0.05,
            numbers_with_elongated_ticks=[0, 0.5]
        ).shift(UP).to_edge(LEFT)
        
        x_line = NumberLine(
            unit_size=12,
            x_min=0,
            x_max=1,
            tick_frequency=0.1,
            numbers_with_elongated_ticks=[0, 1]
        ).shift(DOWN).to_edge(LEFT)
        
        u_label = MathTex("u", color=GREEN)
        u_label.next_to(u_line, RIGHT)
        
        x_label = MathTex("x", color=BLUE)
        x_label.next_to(x_line, RIGHT)
        
        self.play(Create(u_line), Create(u_label))
        self.wait()
        
        self.play(Create(x_line), Create(x_label))
        self.wait()
        
        evenly_spaced_dots = VGroup(*[Dot(u_line.n2p(number)) for number in np.linspace(0, 0.5, num=21)])
        evenly_spaced_dots.set_color_by_gradient(RED, YELLOW)
        
        evenly_spaced_dots.save_state()
        
        self.play(Create(evenly_spaced_dots))
        self.wait()
        
        ghosts = evenly_spaced_dots.copy()
        self.add(ghosts)
        
        self.play(*self.get_dot_mapping_animation(evenly_spaced_dots, lambda x : 2 * x, u_line, x_line))
        self.wait()
        
        linear_text = Tex(r"Linear ", r"$\Leftrightarrow$", r" evenly spaced dots remain evenly spaced")
        linear_text[0].set_color(YELLOW)
        linear_text[2].set_color(YELLOW)
        linear_text.next_to(x_line, DOWN).shift(DOWN)
        
        self.play(Write(linear_text))
        self.wait()
        
        obvious_text = Tex(r"Width ", r"$\longrightarrow$", r" $2($", r"Width", r"$)$", r" $ = $ ", r'Correct Width')
        obvious_text[0].set_color(GREEN)
        obvious_text[3].set_color(GREEN)
        obvious_text[-1].set_color(BLUE)
        
        obvious_text.next_to(transform_text, DOWN)
        self.play(Write(obvious_text[:-2]))
        self.wait()
        self.play(Write(obvious_text[-2:]))
        self.wait()
        
        obvious_math_text = MathTex(r"\Delta u", r"\longrightarrow", r"2", r"\Delta u", r"=", r"\Delta x", r"= \text{Correct Width}")
        obvious_math_text[0].set_color(GREEN)
        obvious_math_text[2].set_color(YELLOW)
        obvious_math_text[3].set_color(GREEN)
        obvious_math_text[5:].set_color(BLUE)
        
        obvious_math_text.next_to(transform_text, DOWN)
        
        self.play(ReplacementTransform(obvious_text, obvious_math_text))
        self.wait()
        
        obvious_scaling_brace = Brace(obvious_math_text[2], DOWN)
        obvious_scaling_brace_text = obvious_scaling_brace.get_text("Multiply by correction factor")
        
        self.play(Create(obvious_scaling_brace), Write(obvious_scaling_brace_text))
        self.wait()
        
        self.play(
            FadeOut(obvious_scaling_brace),
            FadeOut(obvious_scaling_brace_text),
            FadeOut(obvious_math_text),
            FadeOut(transform_text),
            FadeOut(ghosts),
            Restore(evenly_spaced_dots)
        )
        self.play(FadeOut(evenly_spaced_dots))
        self.wait()
        
        new_u_line = NumberLine(
            unit_size=12,
            x_min=0,
            x_max=1,
            tick_frequency=0.1,
            numbers_with_elongated_ticks=[0, 1]
        ).shift(UP).to_edge(LEFT)
        
        self.play(ReplacementTransform(u_line, new_u_line), u_label.animate.next_to(new_u_line, RIGHT))
        self.wait()
        
        u_line = new_u_line
        
        transform_text = MathTex(r"x", r"=", r"u^2").scale(1.5)
        transform_text[0].set_color(BLUE)
        transform_text[2].set_color(GREEN)
        transform_text.to_edge(UP)
        
        self.play(Write(transform_text))
        self.wait()
        
        evenly_spaced_dots = VGroup(*[Dot(u_line.n2p(number)) for number in np.linspace(0, 1.0, num=21)])
        evenly_spaced_dots.set_color_by_gradient(RED, YELLOW)
        
        evenly_spaced_dots.save_state()
        
        self.play(Create(evenly_spaced_dots))
        self.wait()
        
        ghosts = evenly_spaced_dots.copy()
        self.add(ghosts)
        
        self.play(*self.get_dot_mapping_animation(evenly_spaced_dots, lambda x : x * x, u_line, x_line))
        self.wait()
        
        nonlinear_text = Tex(r"Nonlinear").scale(1.5)
        nonlinear_text.next_to(transform_text, DOWN)
        rainbowify(nonlinear_text[0])
        
        self.play(Write(nonlinear_text))
        self.wait()
        
        self.play(FadeOut(ghosts), Restore(evenly_spaced_dots))
        self.wait()
        
        # Fade out the dot corresponding to 0.75
        
        self.play(FadeOut(evenly_spaced_dots[15]))
        new_list = list(evenly_spaced_dots)
        new_list.pop(15)
        evenly_spaced_dots = VGroup(*new_list)
        
        # Now need to cover up the number line, otherwise when we zoom in it gets too large
        # 0.15 here is the zoom factor, 3 is the side length of the display square
        # So 3 * 0.15 should be side length of zoom square
        
        covering_rectangle = Rectangle(
            color=BLACK, height = 3 * 0.15, width = 3 * 0.15, fill=BLACK, fill_opacity=1
        ).move_to(u_line.n2p(0.75))
        self.bring_to_front(covering_rectangle)
        self.add(covering_rectangle)
        
        # Then, we add a dot cluster near the point
        # We need to get the left and right endpoints
        # So, we get the point corresponding to 0.75
        # Subtract off half the width of the camera frame
        # And get the corresponding point on the number line
        
        left_endpoint = u_line.p2n(u_line.n2p(0.75) - np.array([1.5 * 0.15, 0, 0]))
        right_endpoint = u_line.p2n(u_line.n2p(0.75) + np.array([1.5 * 0.15, 0, 0]))
        
        dot_cluster = VGroup(*[Dot(u_line.n2p(number), radius=0.01) for number in np.linspace(left_endpoint, right_endpoint, num=11)])
        dot_cluster.set_color_by_gradient(RED, YELLOW)
        
        cluster_spacing = dot_cluster[1].get_center() - dot_cluster[0].get_center()
        
        center_dot = dot_cluster[5]
        
        # But also need a miniature number line for this
        
        mini_line = Line(u_line.n2p(left_endpoint), u_line.n2p(right_endpoint), stroke_width = 0.5, color=WHITE)
        self.add(mini_line)
        
        
        # Also need to add ticks
        
        ticks = list()
        
        for dot in dot_cluster:
            ticks.append(Line(dot.get_center() - np.array([0, 0.1 * 0.15, 0]), dot.get_center() + np.array([0, 0.1 * 0.15, 0]), stroke_width=0.5))
        
        ticks = VGroup(*ticks)
        self.add(ticks)
        
        center_mark = DecimalNumber(number=0.75).move_to(ticks[5].get_start()).scale(0.1).shift(DOWN*0.04)
        
        self.add(center_mark)
        
        self.play(Create(dot_cluster))
        
        self.wait()
        
        
        
        self.activate_zooming()
        self.zoom_in_on_input(0.75, u_line)
        self.wait()
        
        self.zoomed_camera.frame.add_updater(lambda m: m.move_to(center_dot.get_center()))
        covering_rectangle.add_updater(lambda m: m.move_to(center_dot.get_center()))
        mini_line.add_updater(lambda m: m.move_to(center_dot.get_center()))
        ticks.add_updater(lambda m: m.move_to(center_dot.get_center()))
        
        self.add(self.zoomed_camera.frame, covering_rectangle, mini_line, ticks) # Must add after adding any updater
        self.bring_to_front(dot_cluster) # To stop them getting covered by covering rectangle
        
        
        self.play(
            *self.get_dot_mapping_animation(evenly_spaced_dots, lambda x: x**2, u_line, x_line),
            *self.get_dot_mapping_animation(dot_cluster, lambda x: x**2, u_line, x_line)
        )
        self.wait()
        
        scale_text = Tex(r"Scale by ", r"$1.5$").scale(0.9).to_edge(UP).to_edge(RIGHT).align_to(self.zoomed_camera.frame, alignment_vect=LEFT).shift(DOWN*0.2).shift(LEFT*0.2)
        scale_text[1].set_color(YELLOW)
        self.add_foreground_mobject(scale_text)
        self.play(Write(scale_text))
        self.wait()
        
        onepointfive_copy = scale_text[1].copy()
        
        scale_math_text = MathTex(r"\frac{dx}{du}(0.75) = ", r"1.5").to_edge(UP).to_edge(LEFT)
        scale_math_text[1].set_color(YELLOW)
        self.add_foreground_mobject(onepointfive_copy)
        self.play(ReplacementTransform(onepointfive_copy, scale_math_text[1]), Write(scale_math_text[0]))
        self.wait()
        
        locally_linear = Tex(r"Locally linear", color=YELLOW).scale(1.5).next_to(transform_text, DOWN)
        self.play(ReplacementTransform(nonlinear_text, locally_linear))
        self.wait()
        
        self.play(FadeOut(scale_math_text), FadeOut(scale_text))
        self.wait()
        
        # Gotta move locally linear out of the way
        
        self.play(locally_linear.animate.to_edge(UP).to_edge(LEFT))
        
        """
        self.play(
            FadeOut(locally_linear),
            FadeOut(self.zoomed_camera.frame),
            FadeOut(self.zoomed_display),
            FadeOut(u_line),
            FadeOut(x_line),
            FadeOut(u_label),
            FadeOut(x_label),
            FadeOut(evenly_spaced_dots),
            FadeOut(dot_cluster),
            FadeOut(mini_line),
            FadeOut(ticks),
            FadeOut(linear_text)
        )
        """
        
        # And gotta get rid of camera stuff
        
        self.zoom_activated = False
        
        self.play(
            FadeOut(self.zoomed_camera.frame),
            FadeOut(self.zoomed_display),
        )
        self.wait()
        
        nonobvious_text = Tex(r"Width ", r"$\longrightarrow$", r" $2(0.75)$", r"$($", r"Width", r"$)$ ", r"$ = $", r' Correct Width')
        nonobvious_text[0].set_color(GREEN)
        nonobvious_text[2].set_color(GREEN)
        nonobvious_text[-1].set_color(BLUE)
        
        
        nonobvious_text.next_to(transform_text, DOWN)
        general_scale_factor = MathTex(r"2u").set_color(GREEN).move_to(nonobvious_text[2].get_center())
        self.play(Write(nonobvious_text[:-2]))
        self.wait()
        self.play(Write(nonobvious_text[-2:]))
        self.wait()
        self.play(Transform(nonobvious_text[2], general_scale_factor))
        self.wait()
        
        nonobvious_math_text = MathTex(r"\Delta u", r"\longrightarrow", r"2u", r"\Delta u", r"=", r"\Delta x", r"= \text{Correct Width}")
        nonobvious_math_text[0].set_color(GREEN)
        nonobvious_math_text[2].set_color(YELLOW)
        nonobvious_math_text[3].set_color(GREEN)
        nonobvious_math_text[5:].set_color(BLUE)
        
        nonobvious_math_text.next_to(transform_text, DOWN)
        
        self.play(ReplacementTransform(nonobvious_text, nonobvious_math_text))
        self.wait()
        
        nonobvious_scaling_brace = Brace(nonobvious_math_text[2], DOWN)
        nonobvious_scaling_brace_text = nonobvious_scaling_brace.get_text("Multiply by correction factor")
        
        nonobvious_scaling_brace_text_2 = nonobvious_scaling_brace.get_text(r"Multiply by ", r"width distortion factor")
        nonobvious_scaling_brace_text_2[1].set_color(YELLOW)
        
        self.play(Create(nonobvious_scaling_brace), Write(nonobvious_scaling_brace_text))
        self.wait()
        self.play(Transform(nonobvious_scaling_brace_text, nonobvious_scaling_brace_text_2))
        self.wait()


class LocalWidthDistortionFactor(Scene):
    def construct(self):
        general_text = Tex(r"In general, have ", r"$x$ ", r"$=$ ", r"$g(u)$").scale(1.5).to_edge(UP)
        general_text[1].set_color(BLUE)
        general_text[3].set_color(GREEN)
        
        self.play(Write(general_text))
        self.wait()
        
        width_transformation_law = MathTex(r"\Delta u", r"\longrightarrow", r"g'(u)", r"\Delta u", r"=", r"\Delta x").scale(2)
        width_transformation_law[0].set_color(GREEN)
        width_transformation_law[2].set_color(YELLOW)
        width_transformation_law[3].set_color(GREEN)
        width_transformation_law[5].set_color(BLUE)
        
        width_brace = Brace(width_transformation_law[0], UP)
        width_brace_text = width_brace.get_text(r"Width").scale(1.8)
        width_brace_text.set_color(GREEN)
        
        correct_width_brace = Brace(width_transformation_law[-1], UP)
        correct_width_brace_text = correct_width_brace.get_text(r"Correct width").set_color(BLUE).scale(1.8)
        
        distort_brace = Brace(width_transformation_law[2], DOWN)
        distort_brace_text = distort_brace.get_text(r"Local width distortion factor").set_color(YELLOW).scale(2)
        
        self.play(Write(width_transformation_law))
        self.wait()
        self.play(Create(width_brace), Write(width_brace_text))
        self.wait()
        self.play(Create(correct_width_brace), Write(correct_width_brace_text))
        self.wait()
        self.play(Create(distort_brace), Write(distort_brace_text))
        self.wait()
        
        fade_grp = VGroup(width_brace, width_brace_text, correct_width_brace, correct_width_brace_text, distort_brace, distort_brace_text)
        
        self.play(FadeOut(fade_grp))
        self.wait()
        
        approx_box = SurroundingRectangle(width_transformation_law[4])
        
        approx_sign = MathTex(r"\approx").scale(2).move_to(width_transformation_law[4].get_center())
        
        width_transformation_law[4].save_state()
        
        self.play(Create(approx_box))
        self.play(Transform(width_transformation_law[4], approx_sign))
        self.wait()
        
        equals_brace = Brace(width_transformation_law[4], DOWN)
        equals_brace_text = equals_brace.get_text(r"Only at finite zoom level")
        
        self.play(Create(equals_brace), Write(equals_brace_text))
        self.wait()
        self.play(FadeOut(equals_brace), FadeOut(equals_brace_text))
        self.wait()
        
        du_symbol = MathTex(r"du", color=GREEN).scale(2).move_to(width_transformation_law[0].get_center())
        
        du_symbol_2 = MathTex(r"du", color=GREEN).scale(2).move_to(width_transformation_law[3].get_center())
        
        dx_symbol = MathTex(r"dx", color=BLUE).scale(2).move_to(width_transformation_law[-1].get_center())
        
        self.play(Transform(width_transformation_law[0], du_symbol), Transform(width_transformation_law[3], du_symbol_2), Transform(width_transformation_law[-1], dx_symbol), Restore(width_transformation_law[4]))
        self.play(FadeOut(approx_box))
        self.wait()
        
        just_text = Tex(r"Just integration by substitution formula!", color=YELLOW).scale(1.5).next_to(general_text, DOWN)
        self.play(Write(just_text))
        self.wait()
        

class ThePlan1(Scene):
    def construct(self):
        title_text = Tex(r"The Plan").scale(2).to_edge(UP)
        
        self.add(title_text)
        
        line_1 = Tex(r"Recap ", r"integration", r" and ", r"change of variables", r" in 1D").shift(UP)
        line_1[1].set_color(YELLOW)
        line_1[3].set_color(YELLOW)
        self.add(line_1)
        
        line_2 = Tex(r"See how to extend this to ", r"2D area integrals")
        line_2[1].set_color(YELLOW)
        self.add(line_2)
        
        line_3 = Tex(r"See how this extends to ", r"(generalised) surface integrals").shift(DOWN)
        line_3[1].set_color(YELLOW)
        self.add(line_3)
        
        box = SurroundingRectangle(line_2)
        self.play(Create(box))
        self.wait()
    

class ChangeOfVariables2D(Scene):
    def construct(self):
        title_text = Tex(r"2D Change of variables").scale(1.6).to_edge(UP)
        
        self.play(Write(title_text))
        self.wait()
        
        heads_up = Tex(r"Will only focus on how ", r"area element", " changes").next_to(title_text, DOWN)
        heads_up[1].set_color(YELLOW)
        
        self.play(Write(heads_up))
        self.wait()
        
        heads_up_2 = Tex(r"(Know how ", r"domain", r" and ", r"integrand", r" change)").next_to(heads_up, DOWN)
        heads_up_2[1].set_color(PURPLE)
        heads_up_2[3].set_color(BLUE)
        
        self.play(Write(heads_up_2))
        self.wait()
        
        sub_text = MathTex(r"\begin{bmatrix} x \\ y \end{bmatrix}", r"=", r"\begin{bmatrix} f_1(u, v) \\ f_2(u, v) \end{bmatrix}").scale(2)
        sub_text[0].set_color(BLUE)
        sub_text[2].set_color(YELLOW)
        
        self.play(Write(sub_text))
        self.wait()
        
        map_text = Tex(r"Mapping from ", r"u-v", r" to ", r"x-y").scale(2).next_to(sub_text, DOWN)
        map_text[1].set_color(YELLOW)
        map_text[3].set_color(BLUE)
        
        self.play(Write(map_text))
        self.wait()
        
        general_text = Tex(r"Generally ", r"nonlinear").scale(2).next_to(map_text, DOWN)
        
        rainbowify(general_text[1])
        
        self.play(Write(general_text))
        self.wait()


class WeirdAndWonderfulNonlinearTransformations(Scene):

    def construct(self):
        grid = NumberPlane()
        self.add(grid)
        self.play(Create(grid))
        self.wait()
        grid.save_state()
        grid.prepare_for_nonlinear_transform()
        
        self.play(grid.animate(run_time=2).apply_function(lambda point: [point[0]**2 - point[1]**2, 4 * point[0] * point[1], 0]))
        self.wait()
        self.play(Restore(grid))
        self.wait()
        
        self.play(grid.animate(run_time=2).apply_function(lambda point: complex_to_R3(np.exp(R3_to_complex(point)))))
        self.wait()
        self.play(Restore(grid))
        self.wait()
        
        self.play(grid.animate.apply_function(lambda point: [point[0] + np.sin(point[1]), point[1] + np.sin(point[0]), 0]))
        self.wait()
        self.play(Restore(grid))
        self.wait()


class LocallyLinear2DTransformation(ZoomedScene):
    def __init__(self, **kwargs):
        ZoomedScene.__init__(
            self,
            zoomed_camera_frame_starting_position=np.array([1.5, -0.5, 0]),
            **kwargs
        )

    def construct(self):
        
        grid = NumberPlane()
        self.add(grid)
        self.wait()
        grid.save_state()
        grid.prepare_for_nonlinear_transform()
        
        linear_text = Tex(r"Linear ", r"$\Leftrightarrow$", r" Gridlines remain parallel and evenly spaced", color=YELLOW).scale(0.7)
        linear_text[1].set_color(WHITE)
        linear_text.to_edge(UP).to_edge(LEFT)
        linear_text.add_background_rectangle()
        self.bring_to_front(linear_text)
        self.play(Create(linear_text))
        self.wait()
        self.play(FadeOut(linear_text))
        
        self.activate_zooming()
        
        extra_vertical_lines = list()
        extra_horizontal_lines = list()
        
        for position in np.linspace(- 1.5 * 0.15 * 2, 1.5 * 0.15 * 2, 11):
            extra_vertical_lines.append(Line([1.5 + position, -5, 0], [1.5 + position, 5, 0], stroke_width=0.2))
            extra_horizontal_lines.append(Line([-8, -0.5 + position, 0], [8, -0.5 + position, 0], stroke_width=0.2))
        
        extra_vertical_lines = VGroup(*extra_vertical_lines)
        extra_horizontal_lines = VGroup(*extra_horizontal_lines)
        self.play(Create(extra_vertical_lines), Create(extra_horizontal_lines))
        
        test_dot = Point([1.5, -0.5, 0], color=WHITE)
        
        grid_group = Group(grid, extra_vertical_lines, extra_horizontal_lines, test_dot)
        
        
        
        num_inserted_curves = 50
        
        for mob in extra_vertical_lines: # Stolen from prepare_for_nonlinear_transform
            num_curves = mob.get_num_curves()
            if num_inserted_curves > num_curves:
                mob.insert_n_curves(num_inserted_curves - num_curves)
        
        for mob in extra_horizontal_lines:
            num_curves = mob.get_num_curves()
            if num_inserted_curves > num_curves:
                mob.insert_n_curves(num_inserted_curves - num_curves)
        
        self.zoomed_camera.frame.add_updater(lambda m: m.move_to(test_dot.get_center()))
        self.add(self.zoomed_camera.frame)
        
        grid_group.save_state()
        
        self.play(
            grid_group.animate.apply_function(lambda point: [point[0] + np.sin(point[1]), point[1] + np.sin(point[0]), 0])
        )
        self.wait()



class LocallyLinear2DTransformation_2(ZoomedScene):
    def __init__(self, **kwargs):
        ZoomedScene.__init__(
            self,
            zoomed_camera_frame_starting_position=np.array([1.5, -0.5, 0]),
            zoom_factor=0.05,
            **kwargs
        )

    def construct(self):
        
        grid = NumberPlane()
        self.add(grid)
        self.wait()
        grid.save_state()
        grid.prepare_for_nonlinear_transform()
        
        linear_text = Tex(r"Linear ", r"$\Leftrightarrow$", r" Gridlines remain parallel and evenly spaced", color=YELLOW).scale(0.7)
        linear_text[1].set_color(WHITE)
        linear_text.to_edge(UP).to_edge(LEFT)
        linear_text.add_background_rectangle()
        self.bring_to_front(linear_text)
        self.play(Create(linear_text))
        self.wait()
        self.play(FadeOut(linear_text))
        
        self.activate_zooming()
        
        extra_vertical_lines = list()
        extra_horizontal_lines = list()
        
        for position in np.linspace(- 1.5 * 0.05 * 2, 1.5 * 0.05 * 2, 11):
            extra_vertical_lines.append(Line([1.5 + position, -5, 0], [1.5 + position, 5, 0], stroke_width=0.2))
            extra_horizontal_lines.append(Line([-8, -0.5 + position, 0], [8, -0.5 + position, 0], stroke_width=0.2))
        
        extra_vertical_lines = VGroup(*extra_vertical_lines)
        extra_horizontal_lines = VGroup(*extra_horizontal_lines)
        covering_rectangle = Rectangle(
            color=BLACK, height = 3 * 0.05, width = 3 * 0.05, fill=BLACK, fill_opacity=1
        ).move_to([1.5, -0.5, 0])
        self.add(covering_rectangle, extra_vertical_lines, extra_horizontal_lines)
        self.play(Create(extra_vertical_lines), Create(extra_horizontal_lines))
        
        test_dot = Point([1.5, -0.5, 0], color=WHITE)
        
        grid_group = Group(grid, extra_vertical_lines, extra_horizontal_lines, test_dot)
        
        
        
        num_inserted_curves = 50
        
        for mob in extra_vertical_lines: # Stolen from prepare_for_nonlinear_transform
            num_curves = mob.get_num_curves()
            if num_inserted_curves > num_curves:
                mob.insert_n_curves(num_inserted_curves - num_curves)
        
        for mob in extra_horizontal_lines:
            num_curves = mob.get_num_curves()
            if num_inserted_curves > num_curves:
                mob.insert_n_curves(num_inserted_curves - num_curves)
        
        grid_group.save_state()
        
        self.zoomed_camera.frame.add_updater(lambda m: m.move_to(test_dot.get_center()))
        covering_rectangle.add_updater(lambda m: m.move_to(test_dot.get_center()))
        self.add(self.zoomed_camera.frame, covering_rectangle)
        self.bring_to_front(extra_vertical_lines)
        self.bring_to_front(extra_horizontal_lines)
        
        
        
        self.play(
            grid_group.animate.apply_function(lambda point: [point[0] + np.sin(point[1]), point[1] + np.sin(point[0]), 0])
        )
        self.wait()



class LocallyLinear2DTransformation_3(ZoomedScene):
    def __init__(self, **kwargs):
        ZoomedScene.__init__(
            self,
            zoomed_camera_frame_starting_position=np.array([1.5, -0.5, 0]),
            zoom_factor=0.01,
            zoomed_camera_config={'default_frame_stroke_width': 0.5},
            **kwargs
        )

    def construct(self):
        
        grid = NumberPlane()
        self.add(grid)
        self.wait()
        grid.save_state()
        grid.prepare_for_nonlinear_transform()
        
        linear_text = Tex(r"Linear ", r"$\Leftrightarrow$", r" Gridlines remain parallel and evenly spaced", color=YELLOW).scale(0.7)
        linear_text[1].set_color(WHITE)
        linear_text.to_edge(UP).to_edge(LEFT)
        linear_text.add_background_rectangle()
        self.bring_to_front(linear_text)
        self.play(Create(linear_text))
        self.wait()
        self.play(FadeOut(linear_text))
        
        self.activate_zooming()
        
        extra_vertical_lines = list()
        extra_horizontal_lines = list()
        
        for position in np.linspace(- 1.5 * 0.01 * 2, 1.5 * 0.01 * 2, 11):
            extra_vertical_lines.append(Line([1.5 + position, -5, 0], [1.5 + position, 5, 0], stroke_width=0.05))
            extra_horizontal_lines.append(Line([-8, -0.5 + position, 0], [8, -0.5 + position, 0], stroke_width=0.05))
        
        extra_vertical_lines = VGroup(*extra_vertical_lines)
        extra_horizontal_lines = VGroup(*extra_horizontal_lines)
        covering_rectangle = Rectangle(
            color=BLACK, height = 3 * 0.01, width = 3 * 0.01, fill=BLACK, fill_opacity=1
        ).move_to([1.5, -0.5, 0])
        self.add(covering_rectangle, extra_vertical_lines, extra_horizontal_lines)
        self.play(Create(extra_vertical_lines), Create(extra_horizontal_lines))
        
        test_dot = Point([1.5, -0.5, 0], color=WHITE)
        
        grid_group = Group(grid, extra_vertical_lines, extra_horizontal_lines, test_dot)
        
        
        
        num_inserted_curves = 100
        
        for mob in extra_vertical_lines: # Stolen from prepare_for_nonlinear_transform
            num_curves = mob.get_num_curves()
            if num_inserted_curves > num_curves:
                mob.insert_n_curves(num_inserted_curves - num_curves)
        
        for mob in extra_horizontal_lines:
            num_curves = mob.get_num_curves()
            if num_inserted_curves > num_curves:
                mob.insert_n_curves(num_inserted_curves - num_curves)
        
        grid_group.save_state()
        
        self.zoomed_camera.frame.add_updater(lambda m: m.move_to(test_dot.get_center()))
        covering_rectangle.add_updater(lambda m: m.move_to(test_dot.get_center()))
        self.add(self.zoomed_camera.frame, covering_rectangle)
        self.bring_to_front(extra_vertical_lines)
        self.bring_to_front(extra_horizontal_lines)
        
        
        
        self.play(
            grid_group.animate.apply_function(lambda point: [point[0] + np.sin(point[1]), point[1] + np.sin(point[0]), 0])
        )
        self.wait()


class MicroscopeLocallyLinear2DTransformation(LinearTransformationScene):
    
    def __init__(self, **kwargs):
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False,
            background_plane_kwargs={
            "color": GREY,
            "axis_config": {
                "stroke_color": GREY,
            },
            "background_line_style": {
                "stroke_color": GREY,
                "stroke_width": 1,
            },
        },
            **kwargs
        )
    
    @staticmethod
    def nonlinear_func(point):
        return np.array([point[0] + np.sin(point[1]), point[1] + np.sin(point[0]), 0])
    
    @staticmethod
    def microscope(func, about_point, scale_factor):
        return lambda point: scale_factor*(func(np.array(about_point) + np.array(point) * 1/scale_factor) - func(about_point))
    
    def play_anim_with_scale_factor(self, scale_factor):
        grid = self.grid
        text_to_display = r"Zoom $\times %i$" % scale_factor
        zoom_text = Tex(text_to_display)
        zoom_text.to_edge(UP).to_edge(RIGHT)
        zoom_text.add_background_rectangle()
        self.add_foreground_mobject(zoom_text)
        self.play(Create(zoom_text))
        self.wait()
        self.play(grid.animate.apply_function(self.microscope(self.nonlinear_func, [1.5, -0.5, 0], scale_factor)))
        self.wait()
        self.play(Restore(grid), FadeOut(zoom_text))
        self.wait()
    
    def construct(self):
        self.wait()
        
        self.grid = self.plane
        grid = self.grid
        self.play(Create(grid))
        self.wait()
        grid.save_state()
        grid.prepare_for_nonlinear_transform()
        
        info_text = Tex(r"Zoomed in near $(1.5, -0.5)$").to_edge(UP).to_edge(LEFT)
        info_text.add_background_rectangle()
        self.add_foreground_mobject(info_text)
        self.play(Write(info_text[1]))
        self.wait()
        self.play(FadeOut(info_text))
        
        linear_text = Tex(r"Linear ", r"$\Leftrightarrow$", r" Gridlines remain parallel and evenly spaced", color=YELLOW).scale(0.7)
        linear_text[1].set_color(WHITE)
        linear_text.to_edge(UP).to_edge(LEFT)
        linear_text.add_background_rectangle()
        self.bring_to_front(linear_text)
        self.play(Create(linear_text))
        self.wait()
        self.play(FadeOut(linear_text))
        
        
        self.play_anim_with_scale_factor(1)
        self.play_anim_with_scale_factor(5)
        self.play_anim_with_scale_factor(25)
        self.play_anim_with_scale_factor(100)
        
        
        zoom_text = Tex(r"Zoom $\times 500$")
        zoom_text.to_edge(UP).to_edge(RIGHT)
        zoom_text.add_background_rectangle()
        self.add_foreground_mobject(zoom_text)
        self.play(Create(zoom_text))
        self.wait()
        self.play(grid.animate.apply_function(self.microscope(self.nonlinear_func, [1.5, -0.5, 0], 500)))
        self.wait()
        
        find_text = Tex(r"Find the matrix?", color=YELLOW).to_edge(LEFT).to_edge(UP)
        find_text.add_background_rectangle()
        self.add_foreground_mobject(find_text)
        self.play(Create(find_text))
        self.wait()
        
        
        follow_text = Tex(r"Follow basis vectors...", r" but too big")
        follow_text[1].set_color(RED)
        follow_text[0].add_background_rectangle()
        follow_text[1].add_background_rectangle()
        follow_text.next_to(find_text, DOWN)
        follow_text.to_edge(LEFT)
        self.add_foreground_mobject(follow_text[0])
        self.play(Create(follow_text[0]))
        self.wait()
        self.add_foreground_mobject(follow_text[1])
        self.play(Create(follow_text[1]))
        self.wait()
        self.play(Restore(grid))
        self.wait()
        
        
        
        big_i_hat = Vector([10, 0, 0], color=GREEN, stroke_width=50)
        big_j_hat = Vector([0, 10, 0], color=RED, stroke_width=50)
        self.play(GrowArrow(big_i_hat, run_time=3), GrowArrow(big_j_hat, run_time=3))
        self.add(big_i_hat, big_j_hat)
        self.wait()
        self.play(FadeOut(big_i_hat), FadeOut(big_j_hat), FadeOut(find_text), FadeOut(follow_text[0]), FadeOut(follow_text[1]))
        self.wait()

        
        small_delta_x = self.add_vector([1, 0, 0], color=GREEN)
        small_delta_y = self.add_vector([0, 1, 0], color=RED)
        
        delta_x_label = Matrix(np.array([[r"\Delta u"], ["0"]]), dtype=object, element_alignment_corner=DOWN, include_background_rectangle=True)
        delta_x_label.set_column_colors(GREEN)
        delta_x_label.next_to(small_delta_x, RIGHT + DOWN)
        self.play(Create(delta_x_label))
        
        delta_y_label = Matrix(np.array([[r"0"], ["\Delta v"]]), dtype=object, element_alignment_corner=DOWN, include_background_rectangle=True)
        delta_y_label.set_column_colors(RED)
        delta_y_label.next_to(small_delta_y, RIGHT + UP)
        self.play(Create(delta_y_label))
        
        exploit_text = Tex(r"Exploit linearity").to_edge(UP).to_edge(LEFT)
        exploit_text.add_background_rectangle()
        self.play(Create(exploit_text))
        self.wait()
        
        linear_eq_1 = MathTex(r"L \left(", r"\begin{bmatrix} \Delta u \\ 0 \end{bmatrix}", r"\right)", r"=", r"\Delta u", r"L \left(", r"\begin{bmatrix} 1 \\ 0 \end{bmatrix}", r"\right)")
        linear_eq_1[0].set_color(BLUE)
        linear_eq_1[1].set_color(GREEN)
        linear_eq_1[2].set_color(BLUE)
        linear_eq_1[4].set_color(GREEN)
        linear_eq_1[5].set_color(BLUE)
        linear_eq_1[7].set_color(BLUE)
        linear_eq_1.add_background_rectangle()
        
        linear_eq_1.next_to(exploit_text, DOWN).to_edge(LEFT)
        self.play(Create(linear_eq_1))
        self.wait()
        
        info_brace = Brace(linear_eq_1, DOWN)
        info_brace_text = info_brace.get_text(r'$L$', r' is "limiting linear transformation"').scale(0.8)
        info_brace_text[0].set_color(BLUE)
        info_brace_text.add_background_rectangle()
        
        self.add(info_brace_text)
        self.play(Create(info_brace), Write(info_brace_text[1:]))
        self.wait()
        self.play(FadeOut(info_brace), FadeOut(info_brace_text))
        self.wait()
        
        linear_eq_2 = MathTex(r"\frac{1}{\Delta u}", r"L \left(", r"\begin{bmatrix} \Delta u \\ 0 \end{bmatrix}", r"\right)", r"=", r"L \left(", r"\begin{bmatrix} 1 \\ 0 \end{bmatrix}", r"\right)")
        linear_eq_2[0].set_color(GREEN)
        linear_eq_2[1].set_color(BLUE)
        linear_eq_2[2].set_color(GREEN)
        linear_eq_2[3].set_color(BLUE)
        linear_eq_2[5].set_color(BLUE)
        linear_eq_2[7].set_color(BLUE)
        
        linear_eq_2.add_background_rectangle()
        
        linear_eq_2.next_to(exploit_text, DOWN).to_edge(LEFT)
        self.play(ReplacementTransform(linear_eq_1, linear_eq_2))
        self.wait()
        self.add_foreground_mobject(linear_eq_2)
        
        transformation_eq = MathTex(r"\begin{bmatrix} u \\ v \end{bmatrix} \to \begin{bmatrix} f_1(u, v) \\ f_2(u, v) \end{bmatrix}", color=YELLOW)
        transformation_eq.next_to(linear_eq_2, DOWN).to_edge(LEFT)
        transformation_eq.add_background_rectangle()
        self.play(Create(transformation_eq))
        self.wait()
        self.add_foreground_mobject(transformation_eq)
        
        self.play(FadeOut(delta_y_label), FadeOut(delta_x_label))
        
        num_inserted_curves = 50
        
        grid_group = Group(grid, small_delta_x, small_delta_y)
        
        num_curves = small_delta_x.get_num_curves()
        if num_inserted_curves > num_curves:
            small_delta_x.insert_n_curves(num_inserted_curves - num_curves)
        
        grid.save_state()
        small_delta_x.save_state()
        small_delta_y.save_state()
        
        self.play(grid_group.animate.apply_function(self.microscope(self.nonlinear_func, [1.5, -0.5, 0], 500)))
        self.wait()
        
        new_delta_x_label = Matrix(np.array([[r"\Delta f_1 \bigr |_v"], [r"\Delta f_2 \bigr |_v"]]), dtype=object, element_alignment_corner=DOWN, include_background_rectangle=True)
        new_delta_x_label.set_column_colors(GREEN)
        new_delta_x_label.next_to(small_delta_x, RIGHT + DOWN)
        self.play(Create(new_delta_x_label))
        self.wait()
        
        new_delta_y_label = Matrix(np.array([[r"\Delta f_1 \bigr |_u"], [r"\Delta f_2 \bigr |_u"]]), dtype=object, element_alignment_corner=DOWN, include_background_rectangle=True)
        new_delta_y_label.set_column_colors(RED)
        new_delta_y_label.next_to(small_delta_y, RIGHT + UP)
        self.play(Create(new_delta_y_label))
        self.wait()
        
        self.play(transformation_eq.animate.shift(DOWN*3)) # Make space for new stuff
        
        temp = new_delta_x_label.copy()
        self.add_foreground_mobject(temp)
        
        self.play(temp.animate.move_to(linear_eq_2[2:5].get_center() + DOWN*0.2), linear_eq_2[2:5].animate.set_opacity(0))
        self.wait()
        
        approx_sign = MathTex(r"\approx").move_to(linear_eq_2[5].get_center()) # Since the transformation is only approximately linear at any zoom level
        
        
        
        
        equalsbox = SurroundingRectangle(linear_eq_2[5], buff=0.1)
        self.add_foreground_mobject(equalsbox)
        self.play(Create(equalsbox))
        self.wait()
        self.add_foreground_mobject(approx_sign)
        self.play(linear_eq_2[5].animate.set_opacity(0), FadeIn(approx_sign))
        approx_brace = Brace(approx_sign, DOWN)
        approx_brace_text = approx_brace.get_text(r"Finite zoom level").next_to(temp, RIGHT).shift(DOWN*1.3).add_background_rectangle()
        self.add_foreground_mobjects(approx_brace, approx_brace_text)
        self.play(Create(approx_brace), Write(approx_brace_text[1]))
        self.wait()
        
        zoom_to_deltax_text = Tex(r"Zoom in more $\longrightarrow$ smaller ", r"$\Delta u$").next_to(transformation_eq, UP).to_edge(LEFT).shift(UP*0.5).add_background_rectangle()
        zoom_to_deltax_text[2].set_color(GREEN)
        self.play(Create(zoom_to_deltax_text))
        self.wait()
        
        exact_vector_x = Matrix(np.array([[r"\frac{\partial f_1}{\partial u}"], [r"\frac{\partial f_2}{\partial u}"]]), dtype=object, include_background_rectangle=True, v_buff=1.3)
        exact_vector_x.set_column_colors(GREEN)
        exact_vector_x.move_to(temp.get_center()).shift(DOWN*0.5)
        self.add_foreground_mobject(exact_vector_x)
        
        self.play(
            temp.animate.set_opacity(0), linear_eq_2[1].animate.set_opacity(0), FadeIn(exact_vector_x), FadeOut(approx_sign), linear_eq_2[5].animate.set_opacity(1), approx_brace.animate.set_opacity(0), approx_brace_text.animate.set_opacity(0)
        )
        self.wait()
        self.play(FadeOut(equalsbox))
        self.wait()
        
        jacob_matrix_input = Matrix(np.array([[r"\Delta u"], [r"\Delta v"]]), dtype=object, include_background_rectangle=True).move_to(transformation_eq).to_edge(LEFT).scale(0.8)
        
        jacob_matrix_input.set_row_colors(GREEN, RED)
        
        jacob_matrix_to = MathTex(r"\to", color=YELLOW).next_to(jacob_matrix_input, RIGHT).scale(0.8)
        
        jacob_matrix = Matrix(np.array([[r"\frac{\partial f_1}{\partial u}", r"\frac{\partial f_1}{\partial v}"], [r"\frac{\partial f_2}{\partial u}", r"\frac{\partial f_2}{\partial v}"]]), dtype=object, include_background_rectangle=True, v_buff=1.3, h_buff=1.3, element_alignment_corner=DOWN).scale(0.8)
        
        jacob_matrix.set_column_colors(GREEN, RED)
        jacob_matrix.next_to(jacob_matrix_to, RIGHT)
        
        jacob_matrix_input_cpy = jacob_matrix_input.copy().next_to(jacob_matrix, RIGHT)
        
        jacob_group = Group(jacob_matrix_input, jacob_matrix_to, jacob_matrix, jacob_matrix_input_cpy)
        jacob_group.add_background_rectangle()
        self.add_foreground_mobject(jacob_group)
        
        self.play(FadeOut(transformation_eq), FadeOut(zoom_to_deltax_text), FadeIn(jacob_group))
        
        
        self.wait()
        
        self.play(Restore(grid), Restore(small_delta_x), Restore(small_delta_y))
        self.wait()
        self.play(
            FadeOut(temp),
            FadeOut(new_delta_x_label),
            FadeOut(new_delta_y_label),
            FadeOut(exploit_text),
            FadeOut(exact_vector_x),
            FadeOut(linear_eq_2),
            jacob_group.animate.to_edge(UP)
        )
        
        actual_transformation_eq = MathTex(r"\begin{bmatrix} u \\ v \end{bmatrix} \to \begin{bmatrix} u + \sin(v) \\ v + \sin(u) \end{bmatrix}", color=YELLOW).next_to(zoom_text, DOWN).to_edge(RIGHT)
        
        actual_transformation_eq.add_background_rectangle()
        self.add_foreground_mobject(actual_transformation_eq)
        self.play(Create(actual_transformation_eq))
        self.wait()
        
        actual_jacob_matrix = Matrix(np.array([[r"1", r"\cos(v)"], [r"\cos(u)", r"1"]]), dtype=object, include_background_rectangle=True, v_buff=1.3, h_buff=1.3, element_alignment_corner=DOWN).scale(0.8)
        
        actual_jacob_matrix.set_column_colors(GREEN, RED)
        
        actual_jacob_matrix.next_to(jacob_matrix, DOWN)
        
        self.play(Create(actual_jacob_matrix))
        self.wait()
        
        actual_numerical_jacob_matrix = DecimalMatrix(np.array([[1, 0.878], [0.071, 1]]), element_to_mobject_config={'num_decimal_places': 3}, include_background_rectangle=True)
        
        actual_numerical_jacob_matrix.set_column_colors(GREEN, RED)
        actual_numerical_jacob_matrix.next_to(actual_jacob_matrix, DOWN)
        
        self.play(Create(actual_numerical_jacob_matrix))
        self.wait()
        
        self.play(
            grid_group.animate.apply_function(self.microscope(self.nonlinear_func, [1.5, -0.5, 0], 500))
        )
        self.wait()
        
        self.play(Restore(grid), Restore(small_delta_x), Restore(small_delta_y))
        self.wait()
        self.play(
            FadeOut(actual_jacob_matrix),
            FadeOut(actual_numerical_jacob_matrix),
            FadeOut(actual_transformation_eq)   
        )
        
        want_text = Tex(r"Want local area distortion factor").scale(0.9).next_to(zoom_text, DOWN).to_edge(RIGHT).add_background_rectangle()
        self.add_foreground_mobject(want_text)
        self.play(Write(want_text[1]))
        self.wait()
        
        self.add_unit_square()
        unit_square = self.square
        
        num_curves = unit_square.get_num_curves()
        if num_inserted_curves > num_curves:
            unit_square.insert_n_curves(num_inserted_curves - num_curves)
        self.wait()
        
        
        self.play(
            grid_group.animate.apply_function(self.microscope(self.nonlinear_func, [1.5, -0.5, 0], 500)),
            unit_square.animate.apply_function(self.microscope(self.nonlinear_func, [1.5, -0.5, 0], 500))
        )
        self.wait()
        
        given_text = Tex(r"Given by determinant", color=YELLOW).next_to(want_text, DOWN).to_edge(RIGHT).add_background_rectangle()
        given_arrow = Arrow(given_text.get_corner(DL), unit_square.get_center())
        self.add_foreground_mobjects(given_text, given_arrow)
        
        self.play(Write(given_text[1]), Create(given_arrow))
        self.wait()
        
        technically_text = Tex(r"Technically absolute value - no orientation").scale(0.5).next_to(given_text, DOWN).to_edge(RIGHT).add_background_rectangle()
        
        self.add_foreground_mobject(technically_text)
        
        self.play(Write(technically_text[1]))
        self.wait()



class LocalAreaDistortionFactor(Scene):
    def construct(self):
        overall_eq = MathTex(r"du dv", r"\longrightarrow", r"\abs{\det J}", r"du dv", r"=", r"dx dy").scale(1.7)
        overall_eq[0].set_color(GREEN)
        overall_eq[2].set_color(YELLOW)
        overall_eq[3].set_color(GREEN)
        overall_eq[5].set_color(BLUE)
        
        self.play(Write(overall_eq))
        self.wait()
        
        area_brace = Brace(overall_eq[0], UP)
        area_brace_text = area_brace.get_text(r"Area").scale(1.8)
        area_brace_text.set_color(GREEN)
        
        correct_area_brace = Brace(overall_eq[-1], UP)
        correct_area_brace_text = correct_area_brace.get_text(r"Correct area").set_color(BLUE).scale(1.8)
        
        distort_brace = Brace(overall_eq[2], DOWN)
        distort_brace_text = distort_brace.get_text(r"Local area distortion factor").set_color(YELLOW).scale(2)
        
        self.play(Create(area_brace), Write(area_brace_text))
        self.wait()
        self.play(Create(correct_area_brace), Write(correct_area_brace_text))
        self.wait()
        self.play(Create(distort_brace), Write(distort_brace_text))
        self.wait()
        
        just_text = Tex(r"Just 2D Change of Variables formula!", color=YELLOW).scale(1.6).to_edge(UP)
        
        self.play(Write(just_text))
        self.wait()
        

class ThePlan2(Scene):
    def construct(self):
        title_text = Tex(r"The Plan").scale(2).to_edge(UP)
        
        self.add(title_text)
        
        line_1 = Tex(r"Recap ", r"integration", r" and ", r"change of variables", r" in 1D").shift(UP)
        line_1[1].set_color(YELLOW)
        line_1[3].set_color(YELLOW)
        self.add(line_1)
        
        line_2 = Tex(r"See how to extend this to ", r"2D area integrals")
        line_2[1].set_color(YELLOW)
        self.add(line_2)
        
        line_3 = Tex(r"See how this extends to ", r"(generalised) surface integrals").shift(DOWN)
        line_3[1].set_color(YELLOW)
        self.add(line_3)
        
        box = SurroundingRectangle(line_3)
        self.play(Create(box))
        self.wait()


class SurfaceIntegral(Scene):
    def construct(self):
        title_text = Tex(r"Surface integrals").scale(2).to_edge(UP)
        self.play(Write(title_text))
        self.wait()
        
        still_text = Tex(r"Still have notion of ", r"area element").next_to(title_text, DOWN)
        still_text[1].set_color(YELLOW)
        
        self.play(Write(still_text))
        self.wait()
        
        area_brace = Brace(still_text[1], DOWN)
        area_brace_text = area_brace.get_tex(r"dS").set_color(YELLOW)
        
        self.play(Create(area_brace), Write(area_brace_text))
        self.wait()
        self.play(FadeOut(area_brace), FadeOut(area_brace_text))
        self.wait()
        
        sub_text = MathTex(r"\begin{bmatrix} x \\ y \\ z \end{bmatrix}", r"=", r"\begin{bmatrix} f_1(u, v) \\ f_2(u, v) \\ f_3(u, v) \end{bmatrix}").scale(1.7)
        sub_text[0].set_color(BLUE)
        sub_text[2].set_color(YELLOW)
        
        self.play(Write(sub_text))
        self.wait()
        
        map_text = Tex(r"Mapping from ", r"u-v", r" to ", r"x-y-z").scale(1.7).next_to(sub_text, DOWN)
        map_text[1].set_color(YELLOW)
        map_text[3].set_color(BLUE)
        
        self.play(Write(map_text))
        self.wait()
        
        general_text = Tex(r"Generally ", r"nonlinear").scale(1.7).next_to(map_text, DOWN)
        
        rainbowify(general_text[1])
        
        self.play(Write(general_text))
        self.wait()
        
        

class LocallyLinear2Dto3DTransformation(ThreeDScene):
    
    @staticmethod
    def microscope(func, about_point, scale_factor):
        return lambda point: scale_factor*(func(np.array(about_point) + np.array(point) * 1/scale_factor) - func(about_point))
    
    @staticmethod
    def nonlinear_func(point):
        return np.array([point[0] + np.sin(point[1]), point[1] + np.sin(point[0]), 2 * np.sin(point[0]) + 1 - np.cos(point[1])])
    
    def play_anim_with_scale_factor(self, scale_factor):
        grid = self.grid
        text_to_display = r"Zoom $\times %i$" % scale_factor
        zoom_text = Tex(text_to_display)
        zoom_text.to_edge(UP).to_edge(RIGHT)
        self.add_fixed_in_frame_mobjects(zoom_text)
        self.play(Create(zoom_text))
        self.wait()
        self.play(grid.animate.apply_function(self.microscope(self.nonlinear_func, [1.5, -0.5, 0], scale_factor)))
        self.wait()
        self.play(Restore(grid), FadeOut(zoom_text))
        self.wait()
    
    def construct(self):
        axes = ThreeDAxes(z_min=-4.5, z_max=4.5)
        
        self.add(axes)
        
        self.grid = NumberPlane()
        grid = self.grid
        self.add(grid)
        
        for mobj in [grid, axes]:
            mobj.scale(0.7)
        
        self.set_camera_orientation(phi=60 * DEGREES, theta=-200 * DEGREES)
        self.wait()
        self.begin_ambient_camera_rotation()
        
        linear_text = Tex(r"Linear ", r"$\Leftrightarrow$", r" Gridlines remain parallel and evenly spaced", color=YELLOW).scale(0.7)
        linear_text[1].set_color(WHITE)
        linear_text.to_edge(UP).to_edge(LEFT)
        linear_text.add_background_rectangle()
        self.add_fixed_in_frame_mobjects(linear_text)
        self.play(Create(linear_text))
        self.wait()
        
        grid.prepare_for_nonlinear_transform()
        grid.save_state()
        self.play_anim_with_scale_factor(1)
        self.play_anim_with_scale_factor(5)
        self.play_anim_with_scale_factor(25)
        self.play_anim_with_scale_factor(100)
        self.play_anim_with_scale_factor(500)
        
        # Stolen from self.get_unit_square
        square = Rectangle(
            color=YELLOW,
            width=self.grid.get_x_unit_size(),
            height=self.grid.get_y_unit_size(),
            stroke_color=YELLOW,
            stroke_width=3,
            fill_color=YELLOW,
            fill_opacity=0.3,
        )
        square.move_to(self.grid.coords_to_point(0, 0), DL)
        
        self.play(Create(square))
        
        # Prepare square for nonlinear transform
        
        num_inserted_curves = 50
        
        num_curves = square.get_num_curves()
        if num_inserted_curves > num_curves:
            square.insert_n_curves(num_inserted_curves - num_curves)
        
        grid = self.grid
        grid_group = Group(grid, square)
        grid_group.save_state()
        
        self.play(grid_group.animate.apply_function(self.microscope(self.nonlinear_func, [1.5, -0.5, 0], 500)))
        self.wait()
        
        question_text = Tex(r"Area distortion factor?").next_to(linear_text, DOWN).to_edge(LEFT)
        self.add_fixed_in_frame_mobjects(question_text)
        self.play(Write(question_text))
        self.wait()



class LocallyLinear2Dto3DTransformationAlgebra(Scene):
    def construct(self):
        
        # Stolen from MicroscopeLocallyLinear2DTransformation
        
        jacob_matrix_input = Matrix(np.array([[r"\Delta u"], [r"\Delta v"]]), dtype=object, include_background_rectangle=True)
        
        jacob_matrix_input.set_row_colors(GREEN, RED)
        
        jacob_matrix_to = MathTex(r"\to", color=YELLOW).next_to(jacob_matrix_input, RIGHT)
        
        jacob_matrix = Matrix(np.array([[r"\frac{\partial f_1}{\partial u}", r"\frac{\partial f_1}{\partial v}"], [r"\frac{\partial f_2}{\partial u}", r"\frac{\partial f_2}{\partial v}"], [r"\frac{\partial f_3}{\partial u}", r"\frac{\partial f_3}{\partial v}"]]), dtype=object, include_background_rectangle=True, v_buff=1.3, h_buff=1.3, element_alignment_corner=DOWN)
        
        jacob_matrix.set_column_colors(GREEN, RED)
        jacob_matrix.next_to(jacob_matrix_to, RIGHT)
        
        jacob_matrix_input_cpy = jacob_matrix_input.copy().next_to(jacob_matrix, RIGHT)
        
        jacob_group = VGroup(jacob_matrix_input, jacob_matrix_to, jacob_matrix, jacob_matrix_input_cpy).move_to([0, 0, 0]).to_edge(UP)
        self.play(Create(jacob_group))
        self.wait()
        
        want_brace = Brace(jacob_matrix, DOWN)
        want_brace_text = want_brace.get_text("Want local area distortion factor").set_color(YELLOW)
        
        self.play(Create(want_brace), Write(want_brace_text))
        self.wait()
        
        use_text = Tex(r"Use Gramian!", color=GREEN).scale(2).next_to(want_brace_text, DOWN)
        self.play(Write(use_text))
        self.wait()
        
        self.play(FadeOut(use_text))
        self.wait()
        
        gram_eq = MathTex(r"du", r"dv", r"\longrightarrow", r"\sqrt{\det(\mathbf{J}^T \mathbf{J})}", r"du", r"dv", r"=", r"dS").scale(1.6).next_to(want_brace_text, DOWN)
        gram_eq[0].set_color(GREEN)
        gram_eq[1].set_color(RED)
        gram_eq[3].set_color(YELLOW)
        gram_eq[4].set_color(GREEN)
        gram_eq[5].set_color(RED)
        gram_eq[7].set_color(BLUE)
        
        self.play(Write(gram_eq))
        self.wait()


class ChangeOfVariablesFullAlgebra(Scene):
    def construct(self):
        overall_eq = MathTex(r"\int_S 1 dS = \idotsint_D \sqrt{\det(\mathbf{J}^T \mathbf{J}) } d u_1 \dots d u_m", color=YELLOW).scale(1.5)
        
        self.play(Write(overall_eq))
        self.wait()
        self.play(overall_eq.animate.to_edge(UP))
        
        line_integral = Tex(r"Line integral: $\mathbf{J} = \vec{\mathbf{x} }'(t) \Longrightarrow \int_C 1 ds =$", r"$ \int_a^b \sqrt{\det(\vec{\mathbf{x} }'(t)^T \vec{\mathbf{x} }'(t) )} dt$", color=BLUE).to_edge(LEFT)
        
        self.play(Write(line_integral))
        self.wait()
        
        line_integral_2 = Tex(r"Line integral: $\mathbf{J} = \vec{\mathbf{x} }'(t) \Longrightarrow \int_C 1 ds =$", r"$ \int_a^b \sqrt{\det(\vec{\mathbf{x} }'(t) \cdot \vec{\mathbf{x} }'(t))} dt$", color=BLUE).to_edge(LEFT)
        
        line_integral_3 = Tex(r"Line integral: $\mathbf{J} = \vec{\mathbf{x} }'(t) \Longrightarrow \int_C 1 ds =$", r"$ \int_a^b \sqrt{\vec{\mathbf{x} }'(t) \cdot \vec{\mathbf{x} }'(t)} dt$", color=BLUE).to_edge(LEFT)
        
        line_integral_4 = Tex(r"Line integral: $\mathbf{J} = \vec{\mathbf{x} }'(t) \Longrightarrow \int_C 1 ds =$", r"$ \int_a^b \sqrt{|\vec{\mathbf{x} }'(t)|^2} dt$", color=BLUE).to_edge(LEFT)
        
        line_integral_5 = Tex(r"Line integral: $\mathbf{J} = \vec{\mathbf{x} }'(t) \Longrightarrow \int_C 1 ds =$", r"$ \int_a^b |\vec{\mathbf{x} }'(t)| dt$", color=BLUE).to_edge(LEFT)
        
        self.play(Transform(line_integral[1], line_integral_2[1]))
        self.play(Transform(line_integral[1], line_integral_3[1]))
        self.play(Transform(line_integral[1], line_integral_4[1]))
        self.play(Transform(line_integral[1], line_integral_5[1]))
        self.wait()
        
        surface_integral = Tex(r"Surface integral: $\mathbf{J} = \begin{bmatrix} \frac{\partial \vec{\mathbf{x} } }{\partial u} & \frac{\partial \vec{\mathbf{x} } }{\partial v} \end{bmatrix} \Rightarrow \int_S 1 dS =$", r"$ \iint_D \sqrt{\det(\mathbf{J}^T \mathbf{J})} du dv$", color=PURPLE).scale(0.95).next_to(line_integral, DOWN).to_edge(LEFT)
        
        self.play(Write(surface_integral))
        self.wait()
        
        surface_integral_2 = Tex(r"Surface integral: $\mathbf{J} = \begin{bmatrix} \frac{\partial \vec{\mathbf{x} } }{\partial u} & \frac{\partial \vec{\mathbf{x} } }{\partial v} \end{bmatrix} \Rightarrow \int_S 1 dS =$", r"$ \iint_D |\frac{\partial \vec{\mathbf{x} } }{\partial u} \times \frac{\partial \vec{\mathbf{x} } }{\partial v}| du dv$", color=PURPLE).scale(0.95).next_to(line_integral, DOWN).to_edge(LEFT)
        
        self.play(Transform(surface_integral[1], surface_integral_2[1]))
        self.wait()
        
        change_var = Tex(r"Change of variables: $\mathbf{J}$ is square", r"$\Rightarrow \int_D 1 dx_1 \dots dx_m$", r"$ = \int_D' \sqrt{\det(\mathbf{J}^T \mathbf{J})} du_1 \dots du_m$").arrange(RIGHT).scale(0.75).set_color(GREEN).next_to(surface_integral, DOWN).to_edge(LEFT)
        
        self.play(Write(change_var))
        self.wait()
        
        change_var_2 = Tex(r"$ = \int_D' \sqrt{\det(\mathbf{J}^T) \det(\mathbf{J})} du_1 \dots du_m$").scale(0.6).set_color(GREEN).next_to(change_var[1], RIGHT)
        
        change_var_3 = Tex(r"$ = \int_D' \sqrt{\det(\mathbf{J})^2} du_1 \dots du_m$").scale(0.75).set_color(GREEN).next_to(change_var[1], RIGHT)
        
        change_var_4 = Tex(r"$ = \int_D' |\det(\mathbf{J})| du_1 \dots du_m$").scale(0.75).set_color(GREEN).next_to(change_var[1], RIGHT)
        
        self.play(Transform(change_var[-1], change_var_2))
        self.play(Transform(change_var[-1], change_var_3))
        self.play(Transform(change_var[-1], change_var_4))
        self.wait()
        
        general_text = Tex(r"THE General Formula!", color=YELLOW).scale(2).to_edge(DOWN)
        
        self.play(Write(general_text))
        self.wait()
        
